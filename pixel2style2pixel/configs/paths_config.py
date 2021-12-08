dataset_paths = {
	'celeba_train': '/mnt/pami14/yfyuan/dataset/face/CelebAMask-HQ/CelebA-HQ-img',
	'celeba_test': '/mnt/pami14/yfyuan/dataset/face/CelebAMask-HQ/CelebA-HQ-img',
	'celeba_train_sketch': '',
	'celeba_test_sketch': '',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
    'ffhq': '/mnt/pami23/stma/images256x256',
}

model_paths = {
	'stylegan_ffhq': '/mnt/pami23/stma/pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': '/mnt/pami23/stma/pretrained_models/model_ir_se50.pth',
	'circular_face': '/mnt/pami23/stma/pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': '/mnt/pami23/stma/pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': '/mnt/pami23/stma/pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': '/mnt/pami23/stma/pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': '/mnt/pami23/stma/pretrained_models/moco_v2_800ep_pretrain.pth.tar'
}
