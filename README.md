# GuidedSR-2024
## Environment
- [PyTorch >= 1.10](https://pytorch.org/)

## Training and Testing

**Training with the example option:**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:11342 main.py --scale 8/16 --model_name Net --num_gpus 4 --embed_dim 64 --opt Adam --file_name 'File' --dataset NIR --batch_size 8 --patch_size 256 --loss '1*L1'
```
**Tesing with the example option:**

```bash
python main.py --test_only --load_name 'your_path/model_8/16.pth' --scale 8/16
```

**The pre-trained models and test dataset can be dowoload at:**

- [8 x SR](https://pan.quark.cn/s/83e39ebeccbd)
- [16 x SR](https://pan.quark.cn/s/83e39ebeccbd)
- [Test Dataset]()
