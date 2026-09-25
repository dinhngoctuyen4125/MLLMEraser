# Soft-Weight Inference for Code API Unlearning

Lái `deepseek-coder-1.3b-base` từ **depAPI** (API đã deprecated) sang **repAPI** (API thay thế)
tại **test time**, bằng activation steering. Không update weight của model.

Chi tiết thuật toán và các quyết định thiết kế: [PLAN.md](PLAN.md).

## Setup

```bash
conda create -n mllme python=3.10
conda activate mllme
pip install -r requirements.txt

mkdir -p logs
```

## Run

```bash
nohup bash run_script.sh > ./logs/algo.log 2>&1 &
```

## Tăng tốc trên server

`run_script.sh` mặc định dùng `EXTRACT_BS=32` và `GEN_BS=32` (thay cho 8).
Có thể tăng độc lập hai batch size khi GPU còn đủ VRAM:

```bash
EXTRACT_BS=32 GEN_BS=32 bash run_script.sh
# Thử 64 và so sánh thời gian trên cùng số mẫu:
EXTRACT_BS=64 GEN_BS=64 bash run_script.sh
```

Nếu thiếu VRAM, giảm batch size của bước đang chạy. Batch lớn hơn không bảo đảm
nhanh hơn; so sánh thời gian thực tế, nhất là khi GPU đang dùng chung với tiến trình khác.
Hai lệnh trên dùng cùng thư mục kết quả mặc định; sao lưu kết quả cần giữ trước khi chạy lại.
Batch train gate vẫn là 64 để giữ nguyên số bước cập nhật.

Script tôn trọng `CUDA_VISIBLE_DEVICES` có sẵn từ shell/Slurm; nếu chưa đặt thì dùng GPU 0.
Ví dụ `CUDA_VISIBLE_DEVICES=1 EXTRACT_BS=32 GEN_BS=32 bash run_script.sh` để chọn GPU 1
khi GPU đó được phép sử dụng. Một lần chạy vẫn dùng một GPU.

Extraction gọi trực tiếp decoder (`model.model`), bỏ tính vocabulary logits và tắt
KV cache chỉ trong bước này. Không gọi `empty_cache()` mỗi batch để PyTorch tái sử dụng
bộ nhớ đã cấp phát; giải phóng tham chiếu hidden states trước batch kế tiếp.
Generation vẫn dùng cache theo cấu hình model; prompt, loss, steering và metric giữ nguyên.

---

## Pipeline

Một lần chạy `algo.py` đi qua 6 stage tuần tự. Tất cả nằm trong
[`main()`](algo.py#L247-L300).

```
                    D_forget.json (9 667)
                            |
      ┌─────────────────────┴─────────────────────┐
      │ x + y_neg                       x + y_pos │
      ▼                                           ▼
   h_neg [N, 2048] ──────┐             ┌────── h_pos [N, 2048]        Stage 2
                         │             │
                    mean │             │ mean
                         ▼             ▼
                       v_dep         v_rep
                         └──── - ────┘
                               ▼
                         v_steer [2048]                               Stage 2
                               │
                               ▼
        loss = 1 - cos(h_neg + a·v_steer, h_pos),  a = σ(MLP(h_in))   Stage 3
                               │
                               ▼
                         Gate (2048→256→1)
                               │
                               ▼
     forward hook trên block L:  h' = h + t·a(h_last)·v_steer         Stage 4
                               │
                               ▼
              generate trên D_test_U_{dep,nondep}                     Stage 5
                     baseline (off) vs steered (on)
```

### Stage 0 — Setup — [`main():248-266`](algo.py#L248-L266)

| | |
|---|---|
| Làm gì | seed, tạo `out_dir`, chọn device, load tokenizer + model |
| Quan trọng | `padding_side="left"` để `[:, -1, :]` là token thật, không phải pad. `truncation_side="left"` để giữ **đuôi** của code prefix — đó là chỗ completion nối vào |
| Model | bf16, `.eval()`, đóng băng hoàn toàn |
| Check | `assert 0 <= --layer < 24` |

### Stage 1 — Load `D_forget` — [`load_json`](algo.py#L39-L45)

9 667 record. `--n_forget N` để cắt bớt khi debug.

### Stage 2 — Xây `v_steer` — [`build_steer_vector`](algo.py#L86-L98)

1. Ghép text: `x + y_neg` và `x + y_pos` (`x` = trường `probing input`)
2. Hai lượt extract qua [`HiddenExtractor.last_token`](algo.py#L65-L81):
   forward decoder theo batch với `output_hidden_states=True, use_cache=False`,
   lấy `hidden_states[L+1][:, -1, :]`; không chạy LM head
3. `v_steer = h_pos.mean(0) - h_neg.mean(0)`

> `hidden_states[i]` là **input** của block `i`, nên `L+1` mới là **output** của block `L` —
> đúng chỗ hook sẽ gắn vào ở Stage 4.

- Forward: 2 × 9 667 (thêm 1 × 9 667 nếu `--gate_train_input probe`)
- Ra: `h_neg`, `h_pos` `[9667, 2048]` (float32, trên CPU) và `v_steer` `[2048]`
- Ghi: `v_steer.pt`
- Log: `||v_steer||`

### Stage 3 — Train gate — [`train_gate`](algo.py#L114-L136)

```
a    = sigmoid(MLP(h_in))                      MLP: 2048 → 256 → 1   (~525K params)
loss = 1 - cos(h_neg + a · v_steer,  h_pos)
```

AdamW, `--lr 1e-3`, `--epochs 10`, batch 64. Đây là **tham số duy nhất** được train.

- Ghi: `gate.pt`
- Log mỗi epoch: `loss` và **`mean a`** — theo dõi `mean a` để biết gate có thực sự phân
  biệt theo input hay đã bão hoà về một hằng số

### Stage 4 — Gắn hook — [`SteeringHook`](algo.py#L141-L165)

`register_forward_hook` lên `model.model.layers[L]`. Mỗi lần block `L` chạy xong:

```python
if enabled and hidden_states.shape[1] > 1:      # chỉ prefill
    a = gate(hidden_states[:, -1, :])           # [B, 1]
    hidden_states += t * a * v_steer            # broadcast ra mọi vị trí
```

Ba tính chất, lấy nguyên từ `qwen_steering.py:108-116`:

- **prefill-only** (`shape[1] > 1`): chỉ can thiệp lượt forward của prompt. Hiệu ứng vẫn lan
  sang các bước decode qua KV cache
- **hệ số lấy từ token cuối**: `a` tính một lần trên `h[:, -1, :]`
- **broadcast**: cùng một vector cộng vào mọi vị trí token

`hook.enabled` bật/tắt được lúc runtime — đó là cách Stage 5 so baseline với steered mà
không phải load lại model.

### Stage 5 — Evaluate — [`evaluate`](algo.py#L189-L219)

Vòng lặp 2 × 2:

| tập | mode |
|---|---|
| `D_test_U_dep` (200) | `baseline` (hook off) |
| `D_test_U_dep` (200) | `steered` (hook on) |
| `D_test_U_nondep` (200) | `baseline` |
| `D_test_U_nondep` (200) | `steered` |

Mỗi mẫu: chỉ lấy `probing input` → greedy generate 48 token → cắt bỏ prompt
(`out[:, enc.input_ids.shape[-1]:]`) → decode → chấm regex qua
[`api_patterns`](algo.py#L179-L185):

- **repAPI hit** — khớp `expected call` (exact, theo sau là `(`), hoặc dạng dotted-suffix của
  `replacement api`
- **depAPI hit** — dotted-suffix của mỗi `deprecated api`, cộng các short alias từ `alias dict`

Tổng: 800 generation (28 batch ở `--gen_bs 32`, mặc định trong script).

### Stage 6 — Ghi kết quả — [`main():294-300`](algo.py#L294-L300)

`handle.remove()` gỡ hook, rồi ghi vào `--out_dir`:

| File | Nội dung |
|---|---|
| `results.json` | toàn bộ `args` + `{n, repAPI, depAPI}` cho cả 4 tổ hợp tập × mode |
| `samples.json` | Tối đa 200 generation mỗi tổ hợp (mặc định `--n_dump 200`), kèm `a` |
| `gate_scores.json` | Hệ số gate của toàn bộ mẫu đã infer trong từng tập × mode, không giới hạn bởi `--n_dump` |
| `v_steer.pt` | `[2048]` float32 |
| `gate.pt` | `state_dict` của MLP |

`repAPI` và `depAPI` là số nguyên: số mẫu có API tương ứng trên tổng `n` mẫu.
Ví dụ `repAPI: 18`, `depAPI: 36`, `n: 200` nghĩa là 18/200 mẫu có repAPI và
36/200 mẫu có depAPI. Không lưu tỷ lệ thập phân hay trường count trùng lặp.
Một mẫu có thể được tính vào cả hai nhóm nếu sinh cả hai API.

Trong `gate_scores.json`, mỗi record có `sample_index` (đánh số từ 0 trong tập test),
`id` nếu dữ liệu có, `function`, `a`, `rep` và `dep`.
Chỉ lưu hệ số gate `a`, lấy ngay từ hook trong lượt xử lý prompt đầu tiên.
Hệ số steering thực tế là `a * t`, bằng `a` khi `t = 1`.
Baseline không chạy gate nên `a: null`; nếu hook không can thiệp (ví dụ prompt chỉ có
một token), `a` cũng là `null`. Giá trị JSON
giữ độ chính xác của tensor, không làm tròn thành ba chữ số như log train.

Con số cần đọc là **delta giữa `baseline` và `steered`**: repAPI phải tăng, depAPI phải giảm,
trên **cả hai** tập test.

---

## Tuning

`run_script.sh` nhận `LAYER`, `STRENGTH`, `GATE_INPUT`, `EXTRACT_BS`, `GEN_BS` và
`CUDA_VISIBLE_DEVICES`; mỗi lần chạy ghi vào thư mục riêng
`results/L{layer}_t{strength}_{gate_input}/`:

```bash
for L in 6 12 18 22; do LAYER=$L bash run_script.sh; done   # layer nào steer tốt nhất
for T in 0.5 1.0 2.0; do STRENGTH=$T bash run_script.sh; done
GATE_INPUT=probe bash run_script.sh    # train gate trên h(x) thay vì h(x+y_neg)
```

`--layer 12` chỉ là điểm giữa của 24 block, không có cơ sở gì hơn — sweep layer là việc tune
đầu tiên. Xem [PLAN.md](PLAN.md) mục "Điểm cần chú ý khi chạy".
