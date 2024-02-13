import model_vgg16
import datasets
import train_vgg16


def main():
    save_path = "results/"


    print("dataset is preparing")
    dataset = datasets.Cifar10(target=0)
    print("dataset is prepared","\n")

    print("now, model preparation is started")
    model = model_vgg16.VGG16(classes=dataset.classes)
    print("model preparation is completed","\n")
    
    print("now model building started")
    model.build(input_shape=(None, 32, 32, 3))
    print("model is successfully built!","\n","\n")

    train_vgg16.train_model(model, dataset, save_path)
    model.load_weights(save_path + "ckpt/checkpoints")



if __name__ == '__main__':
    main()