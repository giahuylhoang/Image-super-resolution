from tensorflow.python.keras.layers import Layer
class MetaUpScale(Layer):
    def __init__(self,
                kernel_size = 3,
                hidden_units = 256,
                kernel_initializer = 'he_uniform',
                kernel_regularizer = None,
                kernel_constraint = None,
                **kwargs):
        self.kernel_size = kernel_size
        self.hidden_units = hidden_units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = initializers.get(kernel_regularizer)
        self.kernel_constraint = initializers.get(kernel_constraint)
        super(MetaUpScale, self).__init__(**kwargs)
    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called'
                             'on a list of 2 input [feature maps, scale]')
        if len(input_shape) != 2:
            raise ValueError('This layer should be called'
                             'on a list of 2 input [feature maps, scale]')
        #if len(input_shape[0] > input_shape[1]):
            #raise ValueError('Should be [feature maps, scale]')
        batch_sizes = [s[0] for s in input_shape if s is not None]
        batch_sizes = set(batch_sizes)
        batch_sizes -= set([None])
        if len(batch_sizes) > 1:
            raise ValueError('Cannot deal with tensors with different'
                             'batch sizes. Got tensors with shape :' +
                             str(input_shape))
        #The dimension of output image varies depend on the scale
        #The channel is set to 3 (color image)
        #The number of channel of the feature map
        in_C = input_shape[0][-1]
        out_C = 3
        self.kernel_1 = self.add_weight(name='kernel_fc_1',
                                        shape=(3, self.hidden_units),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        self.kernel_2 = self.add_weight(name='kernel_fc_2',
                                        shape=(self.hidden_units,self.hidden_units),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        self.kernel_3 = self.add_weight(name='kernel_fc_3',
                                        shape=(self.hidden_units, (self.kernel_size ** 2) * in_C * out_C),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        self.bias_1 = self.add_weight(name='bias_fc_1',
                                      shape=(self.hidden_units,),
                                      initializer='zeros')
        self.bias_2 = self.add_weight(name='bias_fc_2',
                                      shape=(self.hidden_units,),
                                      initializer = 'zeros')
        self.bias_3 = self.add_weight(name='bias_fc_3',
                                      shape=((self.kernel_size ** 2) * in_C * out_C,),
                                      initializer='zeros')
        super(MetaUpScale, self).build(input_shape)
                        
    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 input [feature maps, scale]')
        scales = inputs[1]
        features = inputs[0]
        SR_images = list()
        for feature, scale in zip(features, scales):
            in_height, in_width, in_C = tuple(feature.get_shape().as_list())
            out_C = 3
            out_height = int(in_height * scale)
            out_width = int(in_width * scale)
            out_image = list()
            scale = scale.numpy()
            #Padding the edges and corners with zeros
            padd = int(self.kernel_size/2)
            paddings = tf.constant([[padd,padd],
                                    [padd,padd],
                                    [0,0]])
            padded_feature = tf.pad(feature, paddings, 'CONSTANT')
            for i  in range(out_height):
                for j in range(out_width):
                    v_i = i / scale - int(i / scale)
                    v_j = j / scale - int(j / scale)
                    v_tensor = tf.constant([[v_i, v_j, 1 / scale]], dtype=tf.float32)
                    i_prime = int(i / scale)
                    j_prime = int(j / scale)
                    #Predict_weight
                    weights = tf.nn.bias_add(tf.matmul(v_tensor, self.kernel_1), self.bias_1)
                    weights = tf.nn.relu(weights)
                    weights = tf.nn.bias_add(tf.matmul(weights, self.kernel_2), self.bias_2)
                    weights = tf.nn.relu(weights)
                    weights = tf.nn.bias_add(tf.matmul(weights, self.kernel_3), self.bias_3)
                    weights = tf.reshape(weights, [self.kernel_size ,self.kernel_size ,in_C , out_C ])
                    window = tf.expand_dims(padded_feature[i_prime:i_prime + self.kernel_size, 
                                                           j_prime:j_prime + self.kernel_size],-1)
                    #window = tf.map_fn(lambda x: x[i_prime:i_prime + self.kernel_size, 
                    #                               j_prime:j_prime + self.kernel_size], padded_feature, dtype=tf.float64)
                    pixel_value = tf.reduce_sum(tf.math.multiply(weights, window),axis = [0,1,2])
                    out_image.append(pixel_value)
            SR_image = tf.stack(out_image)
        SR_images.append(SR_image)
        return tf.concat(SR_images, axis=0)
    def compute_out_shape(self, input_shape):
        batch_sizes = [s[0] for s in input_shape if s is not None]
        batch_sizes = set(batch_sizes)
        batch_sizes -= set([None])
        if len(batch_sizes) == 1:
            output_shape = (list(batch_sizes)[0],3)
        else:
            output_shape = (None, 3)
        
        return output_shape
