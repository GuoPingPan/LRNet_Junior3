import tensorflow.keras as K
from tensorflow.keras import layers

block_size = 60
DROPOUT_RATE = 0.5
RNN_UNIT = 64


def main():
    model = K.Sequential([
        layers.InputLayer(input_shape=(block_size, 136)),
        layers.Dropout(0.25),
        layers.Bidirectional(layers.GRU(RNN_UNIT)),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(64, activation='relu'),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(2, activation='softmax')
    ],name='video sequence')
    model_diff = K.Sequential([
        layers.InputLayer(input_shape=(block_size - 1, 136)),
        layers.Dropout(0.25),
        layers.Bidirectional(layers.GRU(RNN_UNIT)),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(64, activation='relu'),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(2, activation='softmax')
    ],name='video diff sequence')

    lossFunction = K.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = K.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss=lossFunction,
                  metrics=['accuracy'])
    model_diff.compile(optimizer=optimizer,
                  loss=lossFunction,
                  metrics=['accuracy'])

    print("Loading models and predicting...")
    #----Using Deeperforensics 1.0 Parameters----#
    model.load_weights('./model_weights/deeper/g1.h5')
    model_diff.load_weights('./model_weights/deeper/g2.h5')


    # print("The summary of model:\n")
    # print(model.summary())
    print("The summary of model diff:\n")
    print(model_diff.summary())
    K.utils.plot_model(model,'model.png',show_shapes=True,show_layer_names=False)


if __name__ == "__main__":
    main()