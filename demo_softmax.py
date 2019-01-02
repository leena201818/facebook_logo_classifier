import argparse

import transfer_cnn_softmax_inference
import transfer_cnn_softmax_train
import os

def main(args):
    print(args.mod)
    assert os.path.exists(args.model_file)

    if args.mod == 'predict':
        assert args.pred_dir is not None or args.pred_image is not None or args.pred_dir_generator is not None

        infer = transfer_cnn_softmax_inference.inference(model_file=args.model_file)

        print(args.pred_dir)
        print(args.pred_image)

        if args.pred_dir_generator is not None:
            assert os.path.exists(args.pred_dir_generator)

            pre_result,_,_ = infer.predict_gen(args.pred_dir_generator)
            for file,label in pre_result.items():
                print("{}:{}".format(os.path.basename(file), label))

        if args.pred_dir is not None:
            assert os.path.exists(args.pred_dir)
            pre_result,_ = infer.predict_dir(args.pred_dir)
            for file,label in pre_result.items():
                print("{}:{}".format(os.path.basename(file), label))

        if args.pred_image is not None:
            assert os.path.exists(args.pred_image)
            pre_result,_ = infer.predict([args.pred_image])
            for file,label in pre_result.items():
                print("{}:{}".format(os.path.basename(file), label))

    if args.mod == 'train':
        train = transfer_cnn_softmax_train.train(args.train_dir,args.model_file,epochs=args.epoch,batch_size=args.batch_size,validation_split=args.validation_split,rewrite_feature=True)
        train.train()

    print('transfer completed.')

if __name__ == '__main__':

    # 训练的command arguments: train model_softmax.h5 --train_dir dataset/training_set
    # 预测的command arguments:
    # predict model_softmax.h5 --pred_dir_generator dataset/prediction/
    # --pred_dir
    # dataset / prediction / generator_subdir - -pred_image
    # dataset / prediction / generator_subdir / logo_011304.jpg

    parser = argparse.ArgumentParser()
    parser.add_argument('mod', type=str, help='训练模型or预测',default='predict',choices=['train','predict'])
    parser.add_argument('model_file', type=str, help='模型文件(*.h5)', default='model.h5')
    parser.add_argument('--epoch', type=int, help='训练epoch', default=40)
    parser.add_argument('--batch_size', type=int, help='训练batch_size', default=128)
    parser.add_argument('--train_dir', type=str, help='训练样本目录,子目录名作为分类标签')
    parser.add_argument('--validation_split', type=float, help='验证数据比例', default=0.2)

    parser.add_argument('--pred_dir_generator', type=str, help='待预测图片目录,必须包含子目录，测试图片放置在子目录中')
    parser.add_argument('--pred_dir', type=str, help='待预测图片存放的目录')
    parser.add_argument('--pred_image', type=str, help='测试图片')

    main(parser.parse_args())
