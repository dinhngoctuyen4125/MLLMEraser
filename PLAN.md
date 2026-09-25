# Soft-Weight Inference for Code API Unlearning — Plan

Mục tiêu: ở **test time**, lái `deepseek-coder-1.3b-base` từ **depAPI** (API đã deprecated)
sang **repAPI** (API thay thế), không update weight.

Nền tảng: MLLMEraser (`./MLLMEraser-B41D`, tải từ anonymous repo của paper `mllme.pdf`),
nhưng thay closed-form `W` + null-space bằng một **MLP gate** học được.

---

## Dữ liệu — `data/deepseek/`

| File | N | Có `y_pos`/`y_neg`? | Vai trò |
|---|---|---|---|
| `D_forget.json` | 9 667 | có | train v_steer + gate |
| `D_test_U_dep.json` | 581 | có | test (lấy 200) |
| `D_test_U_nondep.json` | 17 179 | **không** | test (lấy 200) |

Mỗi record:
- `probing input` = prefix code (x)
- `y_neg` = phần tiếp dùng **depAPI** (vd `np.product(...)`)
- `y_pos` = phần tiếp dùng **repAPI** (vd `np.prod(...)`)
- `deprecated api` / `replacement api` / `expected call` / `alias dict` → dùng để chấm điểm

Libraries: pytorch (55%), tensorflow, scipy, seaborn, numpy, sklearn, transformers, pandas.

---

## Thuật toán

### Step 1 — `v_steer`
```
h_neg = last-token hidden(x + y_neg)   tại layer L
h_pos = last-token hidden(x + y_pos)   tại layer L
v_dep = mean(h_neg);  v_rep = mean(h_pos)
v_steer = v_rep - v_dep
```

### Step 2 — Gate
```
a = sigmoid(MLP(h))          MLP: d -> 256 -> 1
loss = 1 - cos(h_neg + a * v_steer, h_pos)
```
MLP là tham số duy nhất được train. Model đóng băng hoàn toàn.

### Step 3 — Inference
Chỉ lấy `probing input`:
```
h'(x) = h(x) + t * a(h(x)) * v_steer     (t = 1)
```
rồi chạy tiếp forward bình thường. Kỳ vọng **cả hai** tập test đều sinh ra repAPI.

---

## Tận dụng code MLLMEraser

| Lấy từ | Dùng vào | Ghi chú |
|---|---|---|
| `Extractor.py:80-106` | `HiddenExtractor.last_token` | batched forward + `hidden_states[L][:, -1, :]` |
| `Extractor.py:39-42` | tokenizer setup | `padding_side="left"` + `pad_token = eos` — bắt buộc để `[:, -1, :]` là token thật |
| `qwen_steering.py:108-116` | `SteeringHook.__call__` | luật inject: prefill-only, hệ số từ token cuối, broadcast mọi vị trí |
| `MLLMU_eval_steering.py:118-119` | `evaluate` | cắt prompt khỏi output rồi decode |

**Không dùng `null_space_util.py`.** Toàn bộ file đó là closed-form
`W = D H_f^T P^T (P H_f H_f^T P^T + γ P P^T)^+` với null-space projection — chính là thứ
MLP gate thay thế. Đưa vào chỉ là code chết.

### Hai điều chỉnh so với bản gốc
1. **Hook thay vì subclass `DecoderLayer`.** Bản gốc copy nguyên `forward()` của
   `Qwen2_5_VLDecoderLayer` nên vỡ mỗi khi transformers đổi signature. Forward hook cho
   cùng kết quả, ít code hơn, không coupling.
2. **Sửa off-by-one.** `hidden_states[i]` là *input* của block i, nhưng repo gốc index
   steering matrix bằng `i` rồi apply *bên trong* block `i`. Ở đây thống nhất:
   `--layer L` → hook block `L`, extract `hidden_states[L+1]`.

---

## Metric

Sinh greedy 48 token từ `probing input`, rồi đếm regex trên phần sinh ra:

- **repAPI rate** — khớp `expected call` (exact, theo sau là `(`), hoặc dạng dotted-suffix
  của `replacement api` (`np.prod(`, `numpy.prod(` — nhưng không khớp `prod(` trần, tránh
  false positive với builtin như `all(`)
- **depAPI rate** — dotted-suffix của mỗi `deprecated api`, cộng các short alias lấy từ
  `alias dict`

Đã kiểm chứng trên 200 mẫu `D_test_U_dep`: detect đúng **200/200** repAPI trên `y_pos` và
**200/200** depAPI trên `y_neg`.

Mỗi tập chạy **2 lần**: `baseline` (hook tắt) và `steered` (hook bật). Con số đáng quan tâm
là delta giữa hai lần.

---

## Điểm cần chú ý khi chạy

1. **Train/infer mismatch của gate.** Gate train trên `h(x + y_neg)` nhưng inference chỉ
   thấy `h(x)`. Đây là mismatch thật, không phải bug. `--gate_train_input probe` train gate
   trên `h(x)` để khử nó — nên sweep cả hai.
2. **Không có negative supervision.** `D_forget` chỉ có cặp forget, không có mẫu nào ép
   `a → 0`. Gate rất có thể hội tụ về `a ≈ 1` cho mọi input, tức là steering input-unaware.
   Với mục tiêu hiện tại (cả 2 tập test đều muốn ra repAPI) thì điều đó chấp nhận được, nhưng
   log `mean a` mỗi epoch để biết gate có thực sự phân biệt hay không. Nếu sau này cần giữ
   utility, dùng trường `retain` trong `D_forget` làm negative.
3. **Chọn layer.** Không có default hiển nhiên. Mặc định `--layer 12` (giữa của 24).
   Sweep layer là việc tune đầu tiên.
4. **Steer chỉ ở prefill** (theo MLLMEraser). Hiệu ứng vẫn lan sang decode qua KV cache.
   Nếu kết quả yếu, đây là chỗ đầu tiên nên thử biến thể (steer cả lúc decode).
5. `D_test_U_nondep.json` nặng 128 MB → vượt giới hạn 100 MB của GitHub, `.gitattributes`
   đã set git-lfs cho `data/**/*.json`. Cần `git lfs install` trước khi clone dùng được.

---

## Sweep đề xuất

```bash
for L in 6 12 18 22; do LAYER=$L bash run_script.sh; done
for T in 0.5 1.0 2.0; do STRENGTH=$T bash run_script.sh; done
GATE_INPUT=probe bash run_script.sh
```

Kết quả ghi ra `results/L{layer}_t{strength}_{gate_input}/{results,samples,v_steer,gate}.*`.
