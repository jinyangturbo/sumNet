from utilities import *

def plot_cifar():
    # resnet-164 1.7M
    resnet_log = np.loadtxt('log/resnet_cifar10/log.csv', delimiter=',')
    resnet_test_acc = resnet_log[:, -1]
    resnet_test_error = (1. - np.array(resnet_test_acc)) * 100
    resnet_train_loss = resnet_log[:, 1]

    # resnet-164-drop 1.7M
    resnet_log_drop = np.loadtxt('log/resnet_cifar10_drop/log.csv', delimiter=',')
    resnet_test_acc_drop = resnet_log_drop[:, -1]
    resnet_test_error_drop = (1. - np.array(resnet_test_acc_drop)) * 100
    resnet_train_loss_drop = resnet_log_drop[:, 1]

    # densenet-100 0.7M
    densenet_log = np.loadtxt('log/densenet_cifar10/log.csv', delimiter=',')
    densenet_test_acc = densenet_log[:, -1]
    densenet_test_error = (1. - np.array(densenet_test_acc)) * 100
    densenet_train_loss = densenet_log[:, 1]

    # densenet-100-drop 0.7M
    densenet_log_drop = np.loadtxt('log/densenet_cifar10_drop/log.csv', delimiter=',')
    densenet_test_acc_drop = densenet_log_drop[:, -1]
    densenet_test_error_drop = (1. - np.array(densenet_test_acc_drop)) * 100
    densenet_train_loss_drop = densenet_log_drop[:, 1]

    # error plot
    lines = [resnet_test_error, resnet_test_error_drop, densenet_test_error, densenet_test_error_drop]
    shapes = ['-', '-', '-', '-']
    colors = ['b', 'r', 'g', 'm']
    labels = ['Test error: ResNet-164 (1.7M)',
              'Test error: ResNet-164-drop (1.7M)',
              'Test error: DenseNet-100 (0.7M)',
              'Test error: DenseNet-100-drop (0.7M)',]
    plot_learning_curve(lines, shapes, colors, labels,
                        save_path='fig/cifar10_error.png',
                        xlabel='epoch', ylabel='test error (%)',
                        xlim=[0, 300], xticks=np.arange(0, 301, 50),
                        ylim=[4, 18], yticks=np.arange(4, 19, 2))

    # loss plot
    lines = [resnet_train_loss, resnet_train_loss_drop, densenet_train_loss, densenet_train_loss_drop]
    shapes = ['-', '-', '-', '-']
    colors = ['b', 'r', 'g', 'm']
    labels = ['Test loss: ResNet-164 (1.7M)',
              'Test loss: ResNet-164-drop (1.7M)',
              'Test loss: DenseNet-100 (0.7M)',
              'Test loss: DenseNet-100-drop (0.7M)', ]
    # loss plot
    plot_learning_curve(lines, shapes, colors, labels,
                        save_path='fig/cifar10_loss.png', logy=True,
                        xlabel='epoch', ylabel='test loss (%)',
                        xlim=[0, 300], xticks=np.arange(0, 301, 50),
                        ylim=[1e-1, 10**2.2], yticks=10**np.arange(-1, 0.1, 2.2))

    # two-side plot
    lines = [resnet_test_error, resnet_train_loss,
             resnet_test_error_drop, resnet_train_loss_drop,
             densenet_test_error, densenet_train_loss,
             densenet_test_error_drop, densenet_train_loss_drop]
    shapes = ['-', ':', '-', ':', '-', ':', '-', ':']
    colors = ['b', 'b', 'r', 'r', 'g', 'g', 'm', 'm']
    labels = ['Test error: ResNet-164 (1.7M)', 'Training loss: ResNet-164 (1.7M)',
              'Test error: ResNet-164-drop (1.7M)', 'Training loss: ResNet-164-drop (1.7M)',
              'Test error: DenseNet-100 (0.7M)', 'Training loss: DenseNet-100 (0.7M)',
              'Test error: DenseNet-100-drop (0.7M)', 'Training loss: DenseNet-100-drop (0.7M)']
    

    positions = [0, 1, 0, 1, 0, 1, 0, 1]
    plot_learning_curve_two_side(lines, shapes, colors, labels, positions,
                                 save_path='fig/cifar10_error_loss.png',
                                 xlabel='epoch', ylabel_1='test error (%)', ylabel_2='training loss',
                                 xlim=[0, 300], xticks=np.arange(0, 301, 50),
                                 ylim_1=[4, 16], yticks_1=np.arange(4, 17, 2),
                                 ylim_2=[1e-1, 10**3], yticks_2=10**np.arange(-1, 0.1, 3))

# plot_cifar()

def plot_resnet():
    # resnet-164 1.7M
    resnet_log = np.loadtxt('log/resnet_cifar10/log.csv', delimiter=',')
    resnet_test_acc = resnet_log[:, -1]
    resnet_test_error = (1. - np.array(resnet_test_acc)) * 100
    resnet_train_loss = resnet_log[:, 1]

    # resnet-164-drop 1.7M
    resnet_log_drop = np.loadtxt('log/resnet_cifar10_drop/log.csv', delimiter=',')
    resnet_test_acc_drop = resnet_log_drop[:, -1]
    resnet_test_error_drop = (1. - np.array(resnet_test_acc_drop)) * 100
    resnet_train_loss_drop = resnet_log_drop[:, 1]

    # error plot
    lines = [resnet_test_error, resnet_test_error_drop]
    shapes = ['-', '-']
    colors = ['b', 'r']
    labels = ['Test error: ResNet-164 (1.7M)',
              'Test error: ResNet-164-drop (1.7M)',]
    plot_learning_curve(lines, shapes, colors, labels,
                        save_path='fig/resnet_cifar10_error.png',
                        xlabel='epoch', ylabel='test error (%)',
                        xlim=[0, 300], xticks=np.arange(0, 301, 50),
                        ylim=[4, 18], yticks=np.arange(4, 19, 2))

    # loss plot
    lines = [resnet_train_loss, resnet_train_loss_drop]
    shapes = ['-', '-', '-', '-']
    colors = ['b', 'r', 'g', 'm']
    labels = ['Test loss: ResNet-164 (1.7M)',
              'Test loss: ResNet-164-drop (1.7M)',]
    # loss plot
    plot_learning_curve(lines, shapes, colors, labels,
                        save_path='fig/resnet_cifar10_loss.png', logy=True,
                        xlabel='epoch', ylabel='test loss (%)',
                        xlim=[0, 300], xticks=np.arange(0, 301, 50),
                        ylim=[1e-1, 10 ** 2.2], yticks=10 ** np.arange(-1, 0.1, 2.2))

    # two-side plot
    lines = [resnet_test_error, resnet_train_loss,
             resnet_test_error_drop, resnet_train_loss_drop,]
    shapes = ['-', ':', '-', ':']
    colors = ['b', 'b', 'r', 'r']
    labels = ['Test error: ResNet-164 (1.7M)', 'Training loss: ResNet-164 (1.7M)',
              'Test error: ResNet-164-drop (1.7M)', 'Training loss: ResNet-164-drop (1.7M)',]

    positions = [0, 1, 0, 1, 0, 1, 0, 1]
    plot_learning_curve_two_side(lines, shapes, colors, labels, positions,
                                 save_path='fig/resnet_cifar10_error_loss.png',
                                 xlabel='epoch', ylabel_1='test error (%)', ylabel_2='training loss',
                                 xlim=[0, 300], xticks=np.arange(0, 301, 50),
                                 ylim_1=[4, 16], yticks_1=np.arange(4, 17, 2),
                                 ylim_2=[1e-1, 10 ** 3], yticks_2=10 ** np.arange(-1, 0.1, 3))

# plot_resnet()

def plot_densenet():
    # densenet-100 0.7M
    densenet_log = np.loadtxt('log/densenet_cifar10/log.csv', delimiter=',')
    densenet_test_acc = densenet_log[:, -1]
    densenet_test_error = (1. - np.array(densenet_test_acc)) * 100
    densenet_train_loss = densenet_log[:, 1]

    # densenet-100-drop 0.7M
    densenet_log_drop = np.loadtxt('log/densenet_cifar10_drop/log.csv', delimiter=',')
    densenet_test_acc_drop = densenet_log_drop[:, -1]
    densenet_test_error_drop = (1. - np.array(densenet_test_acc_drop)) * 100
    densenet_train_loss_drop = densenet_log_drop[:, 1]

    # error plot
    lines = [densenet_test_error, densenet_test_error_drop]
    shapes = ['-', '-']
    colors = ['b', 'r']
    labels = ['Test error: DenseNet-100 (0.7M)',
              'Test error: DenseNet-100-drop (0.7M)',]
    plot_learning_curve(lines, shapes, colors, labels,
                        save_path='fig/densenet_cifar10_error.png',
                        xlabel='epoch', ylabel='test error (%)',
                        xlim=[0, 300], xticks=np.arange(0, 301, 50),
                        ylim=[4, 18], yticks=np.arange(4, 19, 2))

    # loss plot
    lines = [densenet_train_loss, densenet_train_loss_drop]
    shapes = ['-', '-', '-', '-']
    colors = ['b', 'r', 'r', 'r']
    labels = ['Test loss: DenseNet-100 (0.7M)',
              'Test loss: DenseNet-100-drop (0.7M)',]
    # loss plot
    plot_learning_curve(lines, shapes, colors, labels,
                        save_path='fig/densenet_cifar10_loss.png', logy=True,
                        xlabel='epoch', ylabel='test loss (%)',
                        xlim=[0, 300], xticks=np.arange(0, 301, 50),
                        ylim=[1e-1, 10 ** 2.2], yticks=10 ** np.arange(-1, 0.1, 2.2))

    # two-side plot
    lines = [densenet_test_error, densenet_train_loss,
             densenet_test_error_drop, densenet_train_loss_drop,]
    shapes = ['-', ':', '-', ':']
    colors = ['b', 'b', 'r', 'r']
    labels = ['Test error: DenseNet-100 (0.7M)', 'Training loss: DenseNet-100 (0.7M)',
              'Test error: DenseNet-100-drop (0.7M)', 'Training loss: DenseNet-100-drop (0.7M)',]

    positions = [0, 1, 0, 1, 0, 1, 0, 1]
    plot_learning_curve_two_side(lines, shapes, colors, labels, positions,
                                 save_path='fig/densenet_cifar10_error_loss.png',
                                 xlabel='epoch', ylabel_1='test error (%)', ylabel_2='training loss',
                                 xlim=[0, 300], xticks=np.arange(0, 301, 50),
                                 ylim_1=[4, 16], yticks_1=np.arange(4, 17, 2),
                                 ylim_2=[1e-1, 10 ** 3], yticks_2=10 ** np.arange(-1, 0.1, 3))

# plot_densenet()

def plot_vgg():
    # vgg-164 1.7M
    vgg_log = np.loadtxt('log/vgg_cifar10/log.csv', delimiter=',')
    vgg_test_acc = vgg_log[:, -1]
    vgg_test_error = (1. - np.array(vgg_test_acc)) * 100
    vgg_train_loss = vgg_log[:, 1]

    # vgg-19-drop 20M
    vgg_log_drop = np.loadtxt('log/vgg_cifar10_drop/log.csv', delimiter=',')
    vgg_test_acc_drop = vgg_log_drop[:, -1]
    vgg_test_error_drop = (1. - np.array(vgg_test_acc_drop)) * 100
    vgg_train_loss_drop = vgg_log_drop[:, 1]

    # error plot
    lines = [vgg_test_error, vgg_test_error_drop]
    shapes = ['-', '-']
    colors = ['b', 'r']
    labels = ['Test error: ResNet-19 (20M)',
              'Test error: ResNet-19-drop (20M)',]
    plot_learning_curve(lines, shapes, colors, labels,
                        save_path='fig/vgg_cifar10_error.png',
                        xlabel='epoch', ylabel='test error (%)',
                        xlim=[0, 300], xticks=np.arange(0, 301, 50),
                        ylim=[4, 18], yticks=np.arange(4, 19, 2))

    # loss plot
    lines = [vgg_train_loss, vgg_train_loss_drop]
    shapes = ['-', '-', '-', '-']
    colors = ['b', 'r', 'g', 'm']
    labels = ['Test loss: ResNet-19 (20M)',
              'Test loss: ResNet-19-drop (20M)',]
    # loss plot
    plot_learning_curve(lines, shapes, colors, labels,
                        save_path='fig/vgg_cifar10_loss.png', logy=True,
                        xlabel='epoch', ylabel='test loss (%)',
                        xlim=[0, 300], xticks=np.arange(0, 301, 50),
                        ylim=[1e-1, 10 ** 2.2], yticks=10 ** np.arange(-1, 0.1, 2.2))

    # two-side plot
    lines = [vgg_test_error, vgg_train_loss,
             vgg_test_error_drop, vgg_train_loss_drop,]
    shapes = ['-', ':', '-', ':']
    colors = ['b', 'b', 'r', 'r']
    labels = ['Test error: ResNet-19 (20M)', 'Training loss: ResNet-19 (20M)',
              'Test error: ResNet-19-drop (20M)', 'Training loss: ResNet-19-drop (20M)',]

    positions = [0, 1, 0, 1, 0, 1, 0, 1]
    plot_learning_curve_two_side(lines, shapes, colors, labels, positions,
                                 save_path='fig/vgg_cifar10_error_loss.png',
                                 xlabel='epoch', ylabel_1='test error (%)', ylabel_2='training loss',
                                 xlim=[0, 300], xticks=np.arange(0, 301, 50),
                                 ylim_1=[4, 16], yticks_1=np.arange(4, 17, 2),
                                 ylim_2=[1e-1, 10 ** 3], yticks_2=10 ** np.arange(-1, 0.1, 3))

plot_vgg()

'''
epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc
'''

