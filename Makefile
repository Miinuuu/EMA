	
train_ema :
	CUDA_VISIBLE_DEVICES=0  python3 -m vfi.src.train
eval_ema :
	python3 -m vfi.src.eval