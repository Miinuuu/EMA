vct_train:
	python3 -m vct.src.train
	
vct_eval:
	python3 -m vct.src.eval
	
vfi_train :
	CUDA_VISIBLE_DEVICES=1  python3 -m vfi.src.train
vfi_eval :
	python3 -m vfi.src.eval