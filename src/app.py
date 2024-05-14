import streamlit as st
import cv2 as cv
import time
import os
import datetime
from utils import get_true_annot, get_rbns, annotate
from functions import run_all_tests, validate_accuracy

# Función para cargar los estilos CSS
def load_css(file_name):
    with open(file_name, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# Llamar a la función para cargar los estilos CSS
load_css("../assets/styles.css")

# Configuración de detección de dorsales
bd_configPath = '../content/bib-project/RBNR/custom-yolov4-tiny-detector.cfg'
bd_weightsPath = '../content/bib-project/RBNR/custom-yolov4-tiny-detector_best.weights'
bd_classes = ['bib']

nr_configPath = '../content/bib-project/SVHN/custom-yolov4-tiny-detector.cfg'
nr_weightsPath = '../content/bib-project/SVHN/custom-yolov4-tiny-detector_best.weights'
nr_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

true_color = [66, 245, 108]
color = [66, 245, 108]
dorsal_number = 0
st.title('Detección de dorsales de corredores')
def detect_dorsal():
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
            if output:
                dorsal_number = output[0][0]  # Extraer el número del dorsal
                st.success(f'Número de dorsal: {dorsal_number}')
            else:
                st.warning('No se detectaron dorsales en la imagen.')
            end = time.time()

            st.write(f'Tiempo de ejecución: {round(end - start, 2)} segundos')

            if output is not None:
                for detection in output:
                    img = annotate(img, detection, color)

                img_h, img_w = img.shape[:2]
                resized_img = cv.resize(
                    img, (3 * img_w, 3 * img_h), interpolation=cv.INTER_CUBIC)
                RGB_im = cv.cvtColor(resized_img, cv.COLOR_BGR2RGB)
                
                # Mostrar la imagen con anotaciones
                st.image(RGB_im, caption='Imagen con detecciones',
                        use_column_width=True)
                
                # Obtener la fecha y hora actuales
                now = datetime.datetime.now()
                formatted_time = now.strftime("%d-%m-%y_%H_%M_%S")

                # Formatear el nombre del archivo
                filename = f"{dorsal_number}_{formatted_time}.jpg"
        
            # Guardar la imagen procesada en la carpeta especificada
                save_path = '../usr/validation/Full'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                output_filename = os.path.join(save_path, filename)
                cv.imwrite(output_filename, img)
                
                st.success(f'Imagen guardada en: {output_filename}')
                
            else:
                st.warning('No se detectaron dorsales en la imagen cargada.')

    else:
        st.warning('Por favor, carga una imagen para comenzar la detección.')


# Botón para ejecutar la detección
def model_update():
    if st.button('Ejecutar modelo'):
            run_all_tests()
            accuracy = validate_accuracy()
            st.success(f'Detección completada. Precisión: {accuracy:.2f}')

            # # Mostrar la imagen anotada después de la detección
            # annotated_image = cv.imread('/usr/validation/Full/imagen_annot.JPG')
            # st.image(annotated_image, caption='Imagen anotada',
            #          use_column_width=True)

# Sección para ver las imágenes guardadas


def view_saved_images():
    st.header("Galería de Imágenes Guardadas")
    save_path = '../usr/validation/Full'
    saved_images = os.listdir(save_path)
    if not saved_images:
        st.warning("No hay imágenes guardadas.")
    else:
        # Filtrar por fecha o número de dorsal
        search_by = st.radio("Buscar por", ("Fecha", "Número de Dorsal"))
        if search_by == "Fecha":
            search_date = st.date_input("Selecciona una fecha")
            saved_images = [
                img for img in saved_images if search_date.strftime("%d-%m-%y") in img]
        elif search_by == "Número de Dorsal":
            dorsal_number = st.number_input(
                "Ingresa el número de dorsal", min_value=0, max_value=999, step=1)
            saved_images = [
                img for img in saved_images if str(dorsal_number) in img]

        if not saved_images:
            st.warning(
                "No se encontraron imágenes para los criterios de búsqueda seleccionados.")
        else:
            for img in saved_images:
                st.image(os.path.join(save_path, img),
                         caption=img, use_column_width=True)


# Crear una barra de navegación en el sidebar
def create_navbar():
    st.sidebar.title("Menú")
    page = st.sidebar.radio("Ir a", ("Inicio", "Galería de Imágenes", "Actualizar Modelo"))

    if page == "Inicio":
        detect_dorsal()
    elif page == "Galería de Imágenes":
        view_saved_images()
    elif page == "Actualizar Modelo":
        model_update()


# Mostrar la barra de navegación y el contenido de la página
create_navbar()
