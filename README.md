# Presentación del Proyecto: WEB IA

Este repositorio contiene el código fuente de la aplicación creada por Mateo Heras, que utiliza datos de la API [PLATAFORMA de Compras Públicas](https://datosabiertos.compraspublicas.gob.ec/PLATAFORMA/api/search_ocds).

## Descripción del Proyecto

El proyecto se centra en la creación de modelos de detección de anomalías para identificar contratos de compras públicas que se desvían significativamente del promedio en términos del valor acreditado al proveedor. Estos modelos tienen como objetivo mejorar la fiscalización de los contratos, reducir su tiempo de ejecución y optimizar los recursos. Además, los modelos también pueden predecir la posibilidad de que un contrato de compras públicas sea un fraude. Se han desarrollado modelos de aprendizaje automático (ML) específicos para cada tipo de contrato de compras públicas, ya que cada uno presenta sus propias particularidades.

Además de los modelos de ML, se ha creado un Chatbot SQL que se conecta a una base de datos llamada "datos.db", una muestra más pequeña que los datos de entrenamiento, obtenida de la base de datos de Compras Públicas [PLATAFORMA de Compras Públicas](https://datosabiertos.compraspublicas.gob.ec/PLATAFORMA/api/search_ocds). El ChatbotSQL permite acceder a información relevante utilizando lenguaje natural, lo que agiliza la obtención de información para los servidores públicos y reduce la necesidad de solicitar información al equipo de Data Science, mejorando así los tiempos de trabajo.

## Estructura del Repositorio

El repositorio se organiza de la siguiente manera:

- **FINAL**: Contiene los cuadernos de Jupyter utilizados para la descarga, subida y creación de la base de datos en Google Cloud Platform.

- **Google Colab**: Esta carpeta contiene archivos residuales y de respaldo utilizados durante el proyecto, ya que se utilizaron servicios de Google para el entrenamiento de los modelos.

- **Imágenes**: Aquí se encuentran imágenes relevantes utilizadas en el proyecto.

- **Models**: Almacena todos los modelos de ML creados para cada tipo de contrato de compras públicas.

- **Template**: Contiene las plantillas HTML utilizadas en la interfaz web desarrollada con Flask.

## Otros Archivos

- **Preprocessor.pkl**: Este archivo guarda información de preprocesamiento de datos y se utiliza para mantener la coherencia en las transformaciones de datos cuando un usuario ingresa datos desde la interfaz web.

- **app.py**: Contiene la aplicación Flask que impulsa la interfaz web del proyecto.

- **requirements.txt**: Lista las librerías necesarias para ejecutar el proyecto.

- **.env.env**: En caso de que desees replicar el proyecto, es necesario obtener tus propias variables de entorno.

¡Gracias por explorar nuestro proyecto!
