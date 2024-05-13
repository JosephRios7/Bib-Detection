import streamlit as st
import cv2 as cv
import time
from utils import get_true_annot, get_rbns, annotate

# Configuración de detección de dorsales
bd_configPath = '../content/bib-project/RBNR/custom-yolov4-tiny-detector.cfg'
bd_weightsPath = '../content/bib-project/RBNR/custom-yolov4-tiny-detector_best.weights'
bd_classes = ['bib']

nr_configPath = '../content/bib-project/SVHN/custom-yolov4-tiny-detector.cfg'
nr_weightsPath = '../content/bib-project/SVHN/custom-yolov4-tiny-detector_best.weights'
nr_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

true_color = [66, 245, 108]
color = [66, 245, 108]

st.title('Detección de dorsales de corredores')

# Cargar la imagen directamente en el área principal
uploaded_file = st.file_uploader("Cargar imagen", type=['jpg', 'jpeg'])

if uploaded_file is not None:
    # Guardar los datos binarios del archivo cargado en un archivo temporal
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.read())

    # Leer la imagen usando OpenCV
    img = cv.imread("temp_image.jpg")

    # Mostrar la imagen cargada
    st.image(img, caption='Imagen cargada',
             use_column_width=True, channels="BGR")

    # Botón para ejecutar la detección
    if st.button('Ejecutar detección'):
        start = time.time()
        output = get_rbns(img, bd_configPath, bd_weightsPath,
                          bd_classes, nr_configPath, nr_weightsPath, nr_classes)
        end = time.time()

        st.write(f'Pred time: {round(end - start, 2)} seconds')

        if output is not None:
            for detection in output:
                img = annotate(img, detection, color)

            img_h, img_w = img.shape[:2]
            resized_img = cv.resize(
                img, (3 * img_w, 3 * img_h), interpolation=cv.INTER_CUBIC)
            RGB_im = cv.cvtColor(resized_img, cv.COLOR_BGR2RGB)

            st.image(RGB_im, caption='Imagen con detecciones',
                     use_column_width=True)

        else:
            st.warning('No se detectaron dorsales en la imagen cargada.')

else:
    st.warning('Por favor, carga una imagen para comenzar la detección.')


# # Botón para ejecutar la detección
# if st.button('Ejecutar modelo'):
#         run_all_tests()
#         accuracy = validate_accuracy()
#         st.success(f'Detección completada. Precisión: {accuracy:.2f}')

#         # Mostrar la imagen anotada después de la detección
#         annotated_image = cv.imread('/usr/validation/Full/imagen_annot.JPG')
#         st.image(annotated_image, caption='Imagen anotada',
#                  use_column_width=True)
