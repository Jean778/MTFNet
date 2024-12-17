from tensorflow.keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, Flatten, Dense, Dropout, Reshape, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def MTFNet(time_input_shape, envelope_input_shape, detail_input_shape):
    num_classes = 10

    conv1d_filters = [4, 4, 4]
    conv1d_kernels = [1, 3, 5]

    conv2d_filters = [4, 4, 4]
    conv2d_kernels = [(1, 1), (3, 3), (5, 5)]

    dense_units = 256
    dropout_rate = 0.7

    learning_rate = 0.001

    # Define time series input and layers
    time_input = Input(shape=time_input_shape, name="time_input")  # 修改这里
    x_time = time_input
    for filters, kernel_size in zip(conv1d_filters, conv1d_kernels):
        x_time = Conv1D(filters=filters, kernel_size=kernel_size,
                        activation='relu', padding='same')(x_time)
    time_pooled = MaxPooling1D(pool_size=2, name="time_maxpooling")(x_time)
    time_flattened = Flatten(name="time_flattened")(time_pooled)

    # Define envelope input and layers
    envelope_input = Input(shape=envelope_input_shape, name="envelope_input")  # 修改这里
    envelope_reshaped = Reshape((1, envelope_input_shape[0], 1), name="envelope_reshaped")(envelope_input)
    x_envelope = envelope_reshaped
    for filters, kernel_size in zip(conv2d_filters, conv2d_kernels):
        x_envelope = Conv2D(filters=filters, kernel_size=kernel_size,
                            activation='relu', padding='same')(x_envelope)
    envelope_flattened = Flatten(name="envelope_flattened")(x_envelope)

    # Define detail input and layers
    detail_input = Input(shape=detail_input_shape, name="detail_input")  # 修改这里
    detail_reshaped = Reshape((1, detail_input_shape[0], 1), name="detail_reshaped")(detail_input)
    x_detail = detail_reshaped
    for filters, kernel_size in zip(conv2d_filters, conv2d_kernels):
        x_detail = Conv2D(filters=filters, kernel_size=kernel_size,
                          activation='relu', padding='same')(x_detail)
    detail_flattened = Flatten(name="detail_flattened")(x_detail)

    merged_features = concatenate([envelope_flattened, detail_flattened], name="merged_features")

    # Combine all features for classification and regression
    combined_features = concatenate([time_flattened, merged_features], name="combined_features")
    dense_output = Dense(dense_units, activation='relu', name="dense_combined_features")(combined_features)
    dropout_output = Dropout(dropout_rate, name="dropout_combined_features")(dense_output)

    # Define outputs
    gas_output = Dense(num_classes, activation='softmax', name='gas_classification_output')(dropout_output)
    concentration_output = Dense(1, name='gas_concentration_output')(dropout_output)

    # Compile model
    model = Model(inputs=[time_input, envelope_input, detail_input],
                  outputs=[gas_output, concentration_output],
                  name="GasClassificationAndConcentrationModel")

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss={
                      'gas_classification_output': 'sparse_categorical_crossentropy',
                      'gas_concentration_output': 'mean_squared_error'
                  },
                  metrics={
                      'gas_classification_output': 'accuracy',
                      'gas_concentration_output': 'mean_squared_error'
                  })

    return model
