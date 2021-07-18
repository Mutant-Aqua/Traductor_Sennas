import cv2
import mediapipe as mp
import os

#-------------------------------------Almacenamiento del entrenamiento----------------------------------
nombre ='Letra_A'
direccion ='C:/Users/Aqua_Mutant/Documents/Proyectos_Python/Traductor'
carpeta = direccion + '/' + nombre
if not os.path.exists(carpeta):
    print('Carpeta creada: ',carpeta)
    os.makedirs(carpeta)
#-------------------Contador para el nombre de las imagenes
cont = 0

#---------------Lectura de los archivos de la carpeta
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#cap = cv2.VideoCapture(0)

#-----------------------------Objeto que dara seguimiento a las manos
clase_manos = mp.solutions.hands
manos = clase_manos.Hands() #Se deja el primer perimetro en false para evitar que detecte en todo momento
                            #Solo detecta en confianza alta
                            #Segundo parametro: numero maximo de manos
                            #Tercer parametro: confianza minima de deteccion
                            #Cuarto parametro: confianza minima de seguimiento

#----------------------------Metodo para dibujo de manos
dibujo = mp.solutions.drawing_utils #Se dibujan los 21 puntos de las manos

while (1):
    ret,frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = [] #Se almacenan las coordenadas de cada punto
    #print(resultado.multi_hand_landmarks) #Si se quiere ver la existencia de la deteccion

    if resultado.multi_hand_landmarks: #Si existe una referencia en los resultados se abre el if
        for mano in resultado.multi_hand_landmarks: #busca la mano en el listado del descriptor
            for id, lm in enumerate(mano.landmark):#Se obtiuene la iunformacion de la mano encontrada
                #print(id,lm)#Se pasa a pixeles los numericos
                alto, ancho, c = frame.shape # se toma la logintud de las manos
                corx, cory = int(lm.x*ancho), int(lm.y*alto)#Ubicacion de cada punto
                posiciones.append([id,corx,cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
            if len(posiciones) != 0:
                pto_i1 = posiciones[4]
                pto_i2 = posiciones[20]
                pto_i3 = posiciones[12]
                pto_i4 = posiciones[0]
                pto_i5 = posiciones[9]
                x1, y1 = (pto_i5[1]-100),(pto_i5[2]-100)
                ancho, alto = (x1+200),(y1+200)
                x2,y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            dedos_reg = cv2.resize(dedos_reg, (200,200), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(nombre+'/Dedos_{}.jpg'.format(cont),dedos_reg)
            cont = cont + 1

    cv2.imshow("Video",frame)
    k = cv2.waitKey(1)
    if k ==27 or cont >= 300:
        break
cap.release()
cv2.destroyAllWindows()
print('Proceso de lectura "',nombre,'" compleatdo al 100%')