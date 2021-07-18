
import tensorflow
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()

print(tensorflow.config.list_physical_devices('GPU'))

datos_entrenamiento = 'C:/Users/Aqua_Mutant/Documents/Proyectos_Python/Traductor/Clasificacion/Entrenamiento'
datos_validacion = 'C:/Users/Aqua_Mutant/Documents/Proyectos_Python/Traductor/Clasificacion/Validacion'

iteraciones = 20
altura, longitud = 200, 200
batch_size = 1
pasos = 300/1
pasos_validacion = 300/1
filtrosconv1 = 32
filtrosconv2 = 64
filtrosconv3 = 128
tam_filtro1 = (4,4)
tam_filtro2 = (3,3)
tam_filtro3 = (2,2)
tam_pool =(2,2)
clases = 5
lr = 0.0005

preprocesamiento_entre = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip = True
)

preprocesamiento_valido = ImageDataGenerator(
    rescale = 1./255
)

imagen_entrenamiento = preprocesamiento_entre.flow_from_directory(
    datos_entrenamiento,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
)

imagen_validacion = preprocesamiento_valido.flow_from_directory(
    datos_validacion,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
)

cnn = Sequential()

cnn.add(Convolution2D(filtrosconv1, tam_filtro1, padding = 'same', input_shape = (altura, longitud,3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = tam_pool))


cnn.add(Convolution2D(filtrosconv2, tam_filtro2, padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = tam_pool))

cnn.add(Convolution2D(filtrosconv3, tam_filtro3, padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = tam_pool))

cnn.add(Flatten())
cnn.add(Dense(640, activation = 'relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation = 'softmax'))

optimizar = tensorflow.keras.optimizers.Adam(learning_rate = lr)
cnn.compile(loss = 'categorical_crossentropy', optimizer = optimizar, metrics = ['accuracy'])

cnn.fit(imagen_entrenamiento,steps_per_epoch = pasos, epochs = iteraciones, validation_data = imagen_validacion, validation_steps = pasos_validacion)

cnn.save('ModeloVocales.h5')
cnn.save_weights('pesosVocales.h5')