  #!/bin/bash

  echo "========== PHASE 1: PRETRAINING =========="
  caffeinate -i python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml --mask_ratio 0.5
  caffeinate -i python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml --mask_ratio 0.75
  caffeinate -i python pretrain_mae.py --config configs/pretrain_baseline_mps.yaml --mask_ratio 0.9

  echo "========== PHASE 2A: LINEAR PROBING =========="
  for ratio in 0.5 0.75 0.9; do
      python linear_probe.py \
          --checkpoint checkpoints/random_mask_${ratio}/checkpoint_best.pth \
          --output_dir results/random_${ratio}/linear_probe \
          --epochs 100
  done

  echo "========== PHASE 2B: FINE-TUNING =========="
  for ratio in 0.5 0.75 0.9; do
      python finetune.py \
          --checkpoint checkpoints/random_mask_${ratio}/checkpoint_best.pth \
          --output_dir results/random_${ratio}/finetune \
          --epochs 100
  done

  echo "========== PHASE 4: ANALYSIS =========="
  python -m evaluation.cka \
      --checkpoints \
          checkpoints/random_mask_0.5/checkpoint_best.pth \
          checkpoints/random_mask_0.75/checkpoint_best.pth \
          checkpoints/random_mask_0.9/checkpoint_best.pth \
      --labels "0.5" "0.75" "0.9" \
      --output results/analysis/cka_comparison.png

  python -m evaluation.visualize_embeddings \
      --checkpoint checkpoints/random_mask_0.75/checkpoint_best.pth \
      --method tsne \
      --output results/analysis/tsne.png

  echo "========== DONE! =========="
  python results_summary.pyEpoch [99] completed in 81.84s - Avg Loss: 0.8472
Training completed!
========== PHASE 2A: LINEAR PROBING ==========
Using device: cpu
Loading checkpoint from checkpoints/random_mask_0.5/checkpoint_best.pth
Traceback (most recent call last):
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/linear_probe.py", line 337, in <module>
    main()
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/linear_probe.py", line 253, in main
    checkpoint = torch.load(args.checkpoint, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/venv/lib/python3.11/site-packages/torch/serialization.py", line 1529, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default. Please use `torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])` or the `torch.serialization.safe_globals([numpy._core.multiarray.scalar])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
Using device: cpu
Loading checkpoint from checkpoints/random_mask_0.75/checkpoint_best.pth
Traceback (most recent call last):
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/linear_probe.py", line 337, in <module>
    main()
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/linear_probe.py", line 253, in main
    checkpoint = torch.load(args.checkpoint, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/venv/lib/python3.11/site-packages/torch/serialization.py", line 1529, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default. Please use `torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])` or the `torch.serialization.safe_globals([numpy._core.multiarray.scalar])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
Using device: cpu
Loading checkpoint from checkpoints/random_mask_0.9/checkpoint_best.pth
Traceback (most recent call last):
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/linear_probe.py", line 337, in <module>
    main()
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/linear_probe.py", line 253, in main
    checkpoint = torch.load(args.checkpoint, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/venv/lib/python3.11/site-packages/torch/serialization.py", line 1529, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default. Please use `torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])` or the `torch.serialization.safe_globals([numpy._core.multiarray.scalar])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
========== PHASE 2B: FINE-TUNING ==========
Using device: cpu
Loading checkpoint from checkpoints/random_mask_0.5/checkpoint_best.pth
Traceback (most recent call last):
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/finetune.py", line 321, in <module>
    main()
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/finetune.py", line 251, in main
    checkpoint = torch.load(args.checkpoint, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/venv/lib/python3.11/site-packages/torch/serialization.py", line 1529, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default. Please use `torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])` or the `torch.serialization.safe_globals([numpy._core.multiarray.scalar])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
Using device: cpu
Loading checkpoint from checkpoints/random_mask_0.75/checkpoint_best.pth
Traceback (most recent call last):
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/finetune.py", line 321, in <module>
    main()
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/finetune.py", line 251, in main
    checkpoint = torch.load(args.checkpoint, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/venv/lib/python3.11/site-packages/torch/serialization.py", line 1529, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default. Please use `torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])` or the `torch.serialization.safe_globals([numpy._core.multiarray.scalar])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
Using device: cpu
Loading checkpoint from checkpoints/random_mask_0.9/checkpoint_best.pth
Traceback (most recent call last):
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/finetune.py", line 321, in <module>
    main()
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/finetune.py", line 251, in main
    checkpoint = torch.load(args.checkpoint, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/venv/lib/python3.11/site-packages/torch/serialization.py", line 1529, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default. Please use `torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])` or the `torch.serialization.safe_globals([numpy._core.multiarray.scalar])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
========== PHASE 4: ANALYSIS ==========
Loading CIFAR-100 for feature extraction...
100%|███████████████████████████████████████| 169M/169M [00:13<00:00, 12.6MB/s]
Extracting features from all models...

Loading checkpoint: checkpoint_best.pth
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/evaluation/cka.py", line 338, in <module>
    main()
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/evaluation/cka.py", line 316, in main
    cka_matrix = compute_cka_matrix(checkpoints, dataloader, kernel=args.kernel)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/evaluation/cka.py", line 228, in compute_cka_matrix
    checkpoint = torch.load(ckpt_path, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/venv/lib/python3.11/site-packages/torch/serialization.py", line 1529, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default. Please use `torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])` or the `torch.serialization.safe_globals([numpy._core.multiarray.scalar])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
Using device: cpu
Loading checkpoint: checkpoints/random_mask_0.75/checkpoint_best.pth
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/evaluation/visualize_embeddings.py", line 208, in <module>
    main()
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/evaluation/visualize_embeddings.py", line 163, in main
    checkpoint = torch.load(args.checkpoint, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/danieltoberman/Documents/git/Deep_Learning/final_project/venv/lib/python3.11/site-packages/torch/serialization.py", line 1529, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default. Please use `torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])` or the `torch.serialization.safe_globals([numpy._core.multiarray.scalar])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
========== DONE! ==========
/Library/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python: can't open file '/Users/danieltoberman/Documents/git/Deep_Learning/final_project/results_summary.py': [Errno 2] No such file or directory