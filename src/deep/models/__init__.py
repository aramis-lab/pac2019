from .resnet_model import ResNet
from .cnn6layer_model import CNN6Layer
from .inception_model import Inception3D_main


def create_model(options):
    """
    Return model depending on the options given.

    :param options: (argparse.Namespace)
    :return: (nn.Module) corresponding CNN on the correct device (GPU or CPU)
    """

    args = vars(options)
    try:
        model = eval(options.model)(**args)
    except NameError:
        raise NotImplementedError(
            'The model wanted %s has not been implemented in models.' % model_name)

    if options.gpu:
        model.cuda()
    else:
        model.cpu()

    return model
