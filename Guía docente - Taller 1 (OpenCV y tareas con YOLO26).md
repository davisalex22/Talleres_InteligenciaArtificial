# Guía docente — Taller 1: Procesamiento de imágenes con OpenCV y tareas de visión con YOLO26

**Notebook:** `Taller 1_ Procesamiento de imágenes con OpenCV y tareas de visión con YOLO26.ipynb`
**Duración:** 1 h · **Modalidad:** clase virtual, explicando sobre el notebook con la pantalla compartida · **Público:** maestría en IA · Colab gratuito, sin GPU

**Objetivos de aprendizaje** (los mismos de la portada del notebook, "Al terminar podrás"):
1. Aplicar **filtros** con OpenCV y decidir **cuál usar y en qué momento** del proceso.
2. Procesar **video**: detectar movimiento, seguir objetos y contarlos.
3. Detectar **rostros** con OpenCV (primer paso del proyecto final de reconocimiento facial).
4. Usar **YOLO26** para las principales tareas de visión: detectar, segmentar, estimar pose, clasificar y detectar objetos rotados.

**Mensaje central:** "Cada filtro tiene su momento: mejorar contraste → limpiar ruido → analizar → limpiar la máscara. Un sistema puede acertar por casualidad; hay que mirar el proceso, no solo el resultado."

> Todos los valores esperados de esta guía se verificaron ejecutando el notebook. En Colab pueden variar levemente, sobre todo los tiempos.

---

## Cómo leer este guion

La guía sigue **el mismo orden y los mismos títulos que el notebook**. Cada sección de la guía es una sección del notebook, y los subtítulos de la guía coinciden con los subtítulos del notebook. La teoría está dentro de la sección donde el notebook la presenta, así que nunca tienes que saltar a otra parte de la guía.

| Marca | Qué significa |
|---|---|
| **Muestra en pantalla:** | Qué parte del notebook señalar con la pantalla compartida |
| **Haz:** | Acción en el notebook (ejecutar, mover un control, cambiar un valor) |
| **Di:** | Texto para decir en voz alta (adáptalo a tu estilo) |
| **Valores esperados** | Tabla con lo que deben ver al mover cada control |
| **Pregunta al grupo:** | Pregunta para que respondan en el chat o con el micrófono, con la respuesta esperada |
| **Si preguntan:** | Preguntas probables de los alumnos, con respuesta |
| **Transición:** | Frase para pasar a la sección siguiente |

Al final hay apéndices de consulta rápida: plan B técnico, conceptos que se confunden, glosario y material de apoyo.

---

## Antes de clase

- [ ] Ejecutar el notebook completo en una cuenta de Colab gratuita y probar cada control con los valores de esta guía.
- [ ] Compartir el notebook con permiso de **lector**; cada estudiante usa **Archivo → Guardar una copia en Drive**.
- [ ] Probar compartir la pestaña de Colab y subir el zoom del navegador (Ctrl + +) para que los alumnos lean los números y los controles.
- [ ] Repasar las tablas de valores de las Secciones 2, 3, 5 y 6: son las que más se preguntan.
- [ ] Tener la carpeta `material/` (videos) en tu Drive por si GitHub va lento.
- [ ] Pedir la cuenta de Roboflow para el Taller 2.

---

## Estructura de la clase (igual que el notebook)

| Min | Parte del notebook | Sección del notebook | Interactivo |
|---|---|---|---|
| 0–4 | Portada | Título y "Al terminar podrás" | — |
| 0–4 | | **0. Configuración** (se ejecuta mientras hablas de la portada) | — |
| 4–10 | **Parte 1 — OpenCV en acción** | **1. La imagen como datos** (incluye "Caza colores con HSV") | Sí |
| 10–20 | | **2. Filtros: qué hace cada uno** (kernel, galería, 7 casos, kernel propio) | Sí |
| 20–24 | | **3. El orden importa: ruido → filtro → bordes** | Sí |
| 24–30 | | **4. Regiones de interés, máscaras y morfología** | Sí |
| 30–38 | | **5. Video: contar vehículos, sin filtros y con filtros** | Sí |
| 38–43 | **Parte 2 — Visión artificial: rostros y YOLO26** | **6. Detección de rostros con OpenCV (Viola-Jones)** | Sí |
| 43–51 | | **7. Tareas de visión artificial con YOLO26** | — |
| 51–56 | | **8. Aplicación: zona de seguridad con detección + seguimiento** | — |
| 56–60 | Cierre | Lo que te llevas · Próximo taller · Reto | — |
| Casa | Anexo para casa | Análisis geométrico (A, B, C) | Sí |

**Si vas atrasado:** deja "Diseña tu propio kernel" (Sección 2) y el anexo para casa, y en la Sección 7 muestra solo la primera figura. **No recortes** las Secciones 3, 5 y 6: son el centro del taller.

---

## Portada del notebook (min 0–4)

**Muestra en pantalla:** la portada del notebook: el título, la lista **"Al terminar podrás"** con los 4 objetivos y la nota "Funciona en Colab gratuito, sin GPU".

**Haz:** al empezar, pide que abran **su copia** del notebook y pulsen **Entorno de ejecución → Ejecutar todas**. La instalación tarda un minuto y corre mientras hablas (ver Sección 0).

**Di (bienvenida y objetivos):**
> "Hoy vamos a recorrer este notebook de principio a fin. Al terminar van a poder hacer cuatro cosas: elegir qué filtro usar y en qué momento, procesar video para contar objetos, detectar rostros, que es el primer paso del proyecto final, y usar YOLO26 para las principales tareas de visión artificial."

**Di (qué es la visión artificial):**
> "La visión artificial busca que una computadora **extraiga información útil de imágenes o video**: qué hay, dónde está, cuántos son, qué están haciendo. Contar autos en una calle, revisar piezas defectuosas en una fábrica, avisar si alguien entra a una zona peligrosa o desbloquear el celular con la cara: todo eso es una cámara conectada a un programa."
>
> "Hay dos niveles que conviene distinguir. El **procesamiento de imágenes**: entra una imagen y sale otra imagen, mejorada, limpia, con los bordes resaltados. Y la **visión artificial**: entra una imagen y sale **información**, por ejemplo 'hay 2 personas dentro de la zona restringida'."
>
> "Hoy se combinan los dos: el procesamiento clásico prepara los datos y el aprendizaje profundo interpreta. El notebook está ordenado así: la **Parte 1** es OpenCV clásico y la **Parte 2** es visión con inteligencia artificial, con YOLO26."

**Pregunta al grupo:** "Escriban en el chat dónde han visto visión artificial esta semana." (Respuestas típicas: desbloqueo facial del celular, peajes que leen placas, filtros de Instagram, cajas de autoservicio.)

**Si preguntan:**
- **¿Qué es OpenCV?** Una biblioteca libre de visión artificial creada por Intel en 1999 (Gary Bradski), hoy mantenida por la fundación OpenCV.org. Tiene miles de funciones para leer video, filtrar, detectar, medir y dibujar. Licencia Apache 2.
- **¿Qué diferencia hay entre visión artificial y procesamiento de imágenes?** Imagen → imagen mejorada vs imagen → información.

**Transición:**
> "Mientras se termina de instalar, veamos qué está haciendo la configuración."

---

## 0. Configuración (min 0–4, mientras hablas de la portada)

**Haz:** comprueba que todos pulsaron **Ejecutar todas**.

**Di:**
> "La primera celda instala Ultralytics, la biblioteca de YOLO, y tarda alrededor de un minuto. Si la sesión se reinicia sola, no rompieron nada: pasa cuando OpenCV ya estaba cargado de una ejecución anterior. Vuelvan a pulsar **Ejecutar todas**."

**Valores esperados:** "Instalación lista." y luego "Videos listos".

**Si preguntan:**
- **¿Por qué se instala `opencv-contrib`?** OpenCV se distribuye en dos paquetes: el principal y *contrib* (módulos extra). En OpenCV 5 el detector clásico de rostros (`CascadeClassifier`) quedó solo en *contrib*; por eso la celda lo instala con la misma versión que trae Colab.
- **¿De dónde salen los videos?** Del repositorio público `intel-iot-devkit/sample-videos` (licencia CC BY 4.0). Se descargan solos; así no dependemos de Drive.

**Transición:**
> "Empecemos por la pregunta más básica: ¿cómo ve una imagen la computadora?"

---

# Parte 1 — OpenCV en acción

**Muestra en pantalla:** el encabezado **"Parte 1 — OpenCV en acción"**, la imagen de OpenCV y el enlace a la **Plataforma interactiva de OpenCV**.

---

## 1. La imagen como datos (min 4–10)

### La imagen es una tabla de números

**Muestra en pantalla:** la frase del notebook: *"Para la computadora, una imagen es una tabla de números: alto × ancho × 3 canales de color, cada uno entre 0 y 255."*

**Di:**
> "Para la computadora no hay 'una foto de un casco': hay una tabla de números. Cada cuadradito de la imagen es un **píxel**, y cada píxel guarda cuánta luz llegó a ese punto de la cámara."
>
> "Cada número va de **0, negro, a 255, máxima intensidad**. ¿Por qué 255? Porque cada número se guarda en 8 bits, y con 8 bits se pueden contar 256 valores, del 0 al 255. Una imagen en gris tiene un número por píxel; una a color tiene **tres**: azul, verde y rojo."
>
> "Esto explica por qué la visión artificial es difícil. Si mañana hay más luz en la misma escena, el casco es el mismo, pero **todos los números cambian**. Lo mismo pasa con otro ángulo u otra distancia. La computadora tiene que reconocer el mismo objeto a partir de números completamente distintos."

### `img.shape` y el valor de un píxel

**Haz:** ejecuta la celda que imprime la forma de la imagen, un píxel y muestra las dos imágenes ("Correcto" y "Error típico").

**Di (qué es `.shape`):**
> "El notebook toma un cuadro del video de los trabajadores y lo guarda en la variable `img`. Esa variable no es una 'foto': es un **arreglo de NumPy**, una tabla de números. Todo arreglo de NumPy tiene un atributo llamado **`.shape`**, 'forma' en inglés, que dice **cuántos elementos tiene en cada dimensión**. No es una función, por eso va sin paréntesis."
>
> "Aquí `img.shape` es **(540, 960, 3)**. Son tres valores porque una imagen a color es una tabla de **tres dimensiones**:"
>
> "El **primer valor, 540, es el alto**: el número de **filas** de píxeles, de arriba hacia abajo."
>
> "El **segundo, 960, es el ancho**: el número de **columnas**, de izquierda a derecha."
>
> "El **tercero, 3, son los canales de color**: cada píxel guarda **tres números**, uno para el azul, uno para el verde y uno para el rojo, en ese orden en OpenCV."
>
> "Fíjense que el orden es **alto primero y ancho después**, al revés de como solemos decir la resolución, '960 por 540'. Es porque NumPy guarda las imágenes **por filas**, como una matriz: primero la fila, después la columna."

**Di (el píxel que imprime la celda):**
> "La celda también imprime un solo píxel: `img[300, 200]`, fila 300, columna 200. Fíjense que no sale un número, sino **tres**, en orden azul, verde, rojo. Y si multiplicamos 540 × 960 × 3 nos da **1 555 200 números** en una sola imagen. Un video son 30 imágenes como esta por segundo."

**Di (color o gris):**
> "Si la imagen estuviera en escala de grises, `.shape` tendría solo **dos** valores, (540, 960): cada píxel sería un único número, el brillo. **El número de valores de `.shape` nos dice si la imagen es a color o en gris.**"

### Las dos imágenes: "Correcto" y "Error típico: OpenCV guarda en BGR, no en RGB"

**Muestra en pantalla:** las dos imágenes lado a lado. Son **la misma imagen**; la segunda tiene los canales en orden inverso.

**Di:**
> "Miren la imagen de la derecha: los chalecos naranjas se ven **azules**. No es otra foto: son exactamente los mismos números, leídos en otro orden."
>
> "OpenCV guarda cada píxel como **[azul, verde, rojo]**, BGR. Casi todo lo demás, Matplotlib, las páginas web, los editores de fotos, espera **[rojo, verde, azul]**, RGB. Si le damos a Matplotlib la imagen tal cual, toma el primer número, que es el azul, y lo pinta como rojo, y al revés. Por eso lo naranja sale azul y lo azul sale naranja."
>
> "La solución es una línea: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` antes de mostrar. En el notebook la función `mostrar` ya lo hace por ustedes; la imagen de la derecha la dejamos sin convertir a propósito, para que reconozcan el error cuando les pase."

**Di (por qué BGR, si alguien pregunta):**
> "Es herencia histórica. Cuando OpenCV nació, en Intel hacia el año 2000, muchas cámaras y el formato BMP de Windows guardaban los colores en ese orden. OpenCV lo adoptó y lo mantuvo por compatibilidad."

### Caza colores con HSV

**Muestra en pantalla:** el texto del notebook: *"En HSV el color (H, tono) está separado del brillo (V): así podemos quedarnos con un color aunque cambie la luz."* y el **Reto**.

**Di (el problema: en RGB el color y la luz están mezclados):**
> "Queremos encontrar los chalecos naranjas **por su color**. ¿Por qué no lo hacemos directamente con los números BGR? Porque en BGR **el color y la luz están mezclados** en los mismos tres números. Piensen en el mismo chaleco, una parte al sol y otra a la sombra."

| El mismo chaleco naranja | En RGB (rojo, verde, azul) | En HSV de OpenCV (tono, saturación, brillo) |
|---|---|---|
| Al sol | (255, 140, 0) | (**16**, 255, **255**) |
| A la sombra | (128, 70, 0) | (**16**, 255, **128**) |
| ¿Qué cambió? | Cambian el rojo y el verde | **Solo el brillo**; el tono sigue en 16 |

**Di:**
> "En RGB, el sol y la sombra producen números distintos: tendría que adivinar qué rangos cubren todos los naranjas posibles. En HSV, el **tono se queda quieto en 16** y solo cambia el brillo. Eso es lo que dice el notebook con 'el color está separado del brillo': puedo pedir 'tono 16' y lo encuentro con sol o con sombra."

**Di (qué es cada letra, con una analogía):**
> "HSV describe un color como lo describiríamos con palabras. Si les digo 'un naranja intenso y claro', acabo de dar los tres valores:"
>
> "**H, tono (*Hue*):** **qué color es**: naranja, amarillo, verde. Es la posición en una rueda de colores."
>
> "**S, saturación:** **qué tan intenso o puro** es el color. Saturación alta es un naranja fosforescente; saturación baja es un naranja 'lavado', casi gris. Saturación cero es gris: blanco, negro y grises no tienen color."
>
> "**V, brillo (*Value*):** **cuánta luz** tiene. Brillo alto es un color iluminado; brillo bajo, oscuro. Brillo cero es negro, sea del color que sea."

**Di (los valores del tono en OpenCV):**
> "El tono es la posición en una rueda de colores. En OpenCV va de **0 a 179**: el **naranja** está más o menos entre **5 y 20**, el **amarillo** entre **20 y 35**, el verde cerca de 60, el azul cerca de 120, y el rojo en **los dos extremos**, en 0 y en 179. ¿Por qué solo hasta 179 y no hasta 360 grados? Porque OpenCV guarda cada número en un byte, que llega a 255, y 360 no cabe; así que divide los grados entre 2. La saturación y el brillo sí van de 0 a 255."

| Color | H en OpenCV | En este notebook |
|---|---|---|
| Rojo | 0 y 179 (los dos extremos) | — |
| Naranja | ~5 a 20 | Los chalecos |
| Amarillo | ~20 a 35 | Los cascos |
| Verde | ~60 | — |
| Azul | ~120 | — |

**Di (qué hace el control, deslizador por deslizador):**
> "El control tiene **cuatro deslizadores**. Cada uno es una **condición**, y un píxel se marca solo si cumple **las cuatro a la vez**:"

| Deslizador | Rango | Valor inicial | Condición que impone | Si lo subes… | Si lo bajas… |
|---|---|---|---|---|---|
| **Tono mínimo** | 0–179 | 5 | El tono debe ser **≥** este valor | Excluyes colores "más rojos" | Aceptas colores más rojos |
| **Tono máximo** | 0–179 | 20 | El tono debe ser **≤** este valor | Aceptas colores "más amarillos/verdes" | Excluyes colores más amarillos |
| **Saturación mínima** | 0–255 | 150 | El color debe ser **al menos así de intenso** | Solo colores muy vivos | Entran colores lavados (piso, pared) |
| **Brillo mínimo** | 0–255 | 150 | El píxel debe tener **al menos esta luz** | Solo zonas iluminadas | Entran zonas oscuras y sombras |

> "La saturación y el brillo **máximos** quedan fijos en 255: no nos molesta que un color sea demasiado intenso o demasiado claro."
>
> "Con los valores iniciales la regla dice: 'márcame los píxeles con **tono entre 5 y 20** (naranja), **saturación de 150 o más** (naranja vivo) y **brillo de 150 o más** (bien iluminado)'. Esos son los chalecos."

**Di (cómo leer el resultado):**
> "A la izquierda está la **máscara**: blanco = el píxel cumple las cuatro condiciones; negro = no. A la derecha, la imagen con un **recuadro verde** por cada mancha blanca grande, y el título dice cuántos objetos encontró. Por dentro, el código arma la máscara con `inRange`, borra los puntitos sueltos y solo cuenta las manchas de **más de 300 píxeles**."

**Haz:** con los valores iniciales, muestra el resultado.

**Di:**
> "Detecta los chalecos, pero dice **5 objetos** y no 2. Las franjas reflectivas son plateadas, no naranjas, así que **parten** cada chaleco en pedazos. El detector encuentra el color, pero no sabe que son 2 chalecos. Eso lo vamos a arreglar con morfología en la Sección 4."

### Reto: detectar solo los cascos amarillos

**Di (planteamiento):**
> "Reto del notebook: muevan los controles hasta que detecte **solo los 2 cascos amarillos**. Pista: acuérdense de los valores del tono. ¿En qué valor de H está el amarillo? Cuando lo logren, escriban sus valores en el chat."

**Haz:** deja 1 minuto para que prueben en su copia. Luego resuelve en vivo, **en este orden**, explicando cada movimiento:

| Paso | Deslizador | De → a | Por qué | Di |
|---|---|---|---|---|
| 1 | **Tono máximo** | 20 → **35** | El amarillo está en ~30; hay que dejarlo entrar | "Primero abro el rango hacia el amarillo." |
| 2 | **Tono mínimo** | 5 → **20** | El naranja está por debajo de 20; así lo saco | "Ahora cierro por abajo para que el naranja de los chalecos ya no entre." |
| 3 | **Saturación mínima** | 150 → **100** | Da margen a las zonas del casco con menos color (reflejos, curvatura) | "Bajo un poco la exigencia de color intenso para que el casco salga completo." |
| 4 | **Brillo mínimo** | **150** (no se toca) | Los cascos están bien iluminados | "El brillo lo dejo igual." |

**Valores esperados:** con **Tono 20–35, Saturación ≥ 100, Brillo ≥ 150** el título dice **2 objetos**: los 2 cascos amarillos. **Esa es la solución del reto.**

**Di (el orden importa también aquí):**
> "¿Por qué subí primero el máximo y después el mínimo? Porque si subo el mínimo a 20 mientras el máximo sigue en 20, el rango queda en un solo valor, de 20 a 20, y la máscara se queda casi vacía. Abran primero, cierren después."

**Pregunta al grupo:** "¿Qué rompería este detector?" (Respuesta: otro color de chaleco, la noche, un cono naranja, una luz de color.)

**Si preguntan:**
- **¿Por qué los valores van de 0 a 255?** Cada valor se guarda en 8 bits: 2⁸ = 256 niveles. Hay imágenes de 10, 12 o 16 bits (médicas, cámaras profesionales) con más niveles.
- **¿Cómo detecto el color rojo?** Con dos rangos (0–10 y 170–179) unidos con `bitwise_or`, porque el rojo está en los dos extremos de la rueda.
- **¿Y el blanco o el negro?** No tienen tono definido (saturación casi 0): se filtran por saturación y brillo, no por tono.
- **¿Por qué se trabaja tanto en escala de grises?** Un tercio de los datos, y la mayor parte de la información de forma (bordes, texturas) está en el brillo.
- **¿Qué pasa si sumo 10 a un píxel que vale 250?** Con NumPy da 4 (se desborda y da la vuelta: 260 − 256); con `cv2.add`, 255 (satura).

**Transición:**
> "La máscara de los chalecos salió en pedazos y con puntitos. Para arreglar eso necesitamos filtros. Veamos qué hace cada uno."

---

## 2. Filtros: qué hace cada uno (min 10–20)

### Qué es un filtro y qué es un kernel (los 3 GIF)

**Muestra en pantalla:** el texto del notebook: *"Un filtro es una pequeña tabla de números (el kernel) que se desliza sobre la imagen: en cada posición multiplica, suma y escribe el resultado."* y las **3 animaciones** de abajo.

**Di (qué es un kernel):**
> "Para cambiar una imagen, por ejemplo para suavizarla o para encontrar sus bordes, hay que decidir **cómo calcular el nuevo valor de cada punto**. La idea clave es que el nuevo valor de un punto depende de **él mismo y de sus vecinos**."
>
> "El **kernel** es una **tablita de números**, por ejemplo de 3 por 3, que funciona como una **receta**: dice **cuánto importa cada vecino** al calcular el nuevo valor del punto del centro. La palabra viene del inglés, 'núcleo'. También lo van a ver llamado **filtro** o **máscara de convolución**."

**Muestra en pantalla:** la primera animación ("Sin relleno"). Señala el cuadrado sombreado que se mueve sobre la cuadrícula azul.

**Di (cómo se aplica):**
> "Miren la animación. La cuadrícula **azul** es la imagen. El cuadrado **sombreado** que se mueve encima es el **kernel**. La cuadrícula **verde** es la imagen nueva que se va construyendo."
>
> "En cada posición, el kernel mira los puntos que tiene debajo, los combina según su receta y escribe **un solo número** en la imagen verde. Después se mueve un lugar y repite, hasta recorrer toda la imagen. Ese recorrido completo se llama **convolución**."
>
> "Lo poderoso es que el mecanismo es siempre el mismo: **cambiando solo los números del kernel**, cambia lo que hace. Con una receta la imagen se **suaviza**, con otra se **enfoca** y con otra solo quedan los **bordes**. Eso es lo que vamos a ver en la galería."

**Di (las tres animaciones):**
> "**Sin relleno:** el kernel no puede centrarse en los puntos del borde, porque se saldría de la imagen. Por eso la imagen verde queda **más pequeña** que la azul."
>
> "**Con relleno (*padding*):** para no perder tamaño, se agrega un **marco de puntos alrededor** de la imagen. Ahora el kernel sí puede centrarse en el borde y la salida **conserva el tamaño**. OpenCV lo hace automáticamente."
>
> "**Saltando de 2 en 2 (*stride*):** el kernel avanza dos casillas en vez de una, así que se detiene la mitad de veces y la salida queda **a la mitad**. Así reducen el tamaño de la imagen las redes neuronales que veremos con YOLO."

**Si preguntan:**
- **¿Por qué el kernel es de 3×3, 5×5, 7×7 y nunca de 4×4?** Porque necesita un **centro**: el punto cuyo valor se está calculando.
- **¿Qué diferencia hay entre un kernel y una máscara?** El kernel es la tablita de números que recorre la imagen; la máscara (Secciones 1 y 4) es una imagen en blanco y negro que dice **dónde** mirar.
- **¿Qué es la convolución en las redes neuronales?** La misma operación, pero los números del kernel no los elige una persona: **los aprende la red** (lo vemos en la Sección 7).

### La galería: "Así se ven los filtros más usados sobre la misma imagen" (3 min)

**Haz:** con la pantalla compartida, baja a la galería de filtros del notebook. Son **12 imágenes**: la misma foto de los trabajadores (en gris), con un filtro distinto en cada una. Ve señalando cada imagen mientras la explicas.

**Di (para abrir):**
> "Un **filtro** es una operación que **cambia la imagen para resolver un problema**: quitar ruido, ver mejor los detalles, encontrar los contornos o separar un objeto del fondo. Todas estas imágenes son **la misma foto**; lo único que cambia es el filtro."
>
> "No hay un filtro 'mejor'. La pregunta siempre es: **¿qué problema tiene mi imagen y qué quiero hacer después?** Los vamos a ver en cuatro grupos."

**Resumen de la galería (tenlo a la vista mientras explicas):**

| Grupo | Filtro | Qué hace | Cuándo usarlo | En qué momento |
|---|---|---|---|---|
| **1. Quitar ruido** | Desenfoque (promedio) | Difumina la imagen de forma pareja | Casi nunca: el gaussiano hace lo mismo, mejor | Al principio |
| | Gaussiano | Difumina suavemente; quita el grano | Imagen **granulada** (cámara de noche, sensor barato) | Al principio |
| | Mediana | Borra puntos sueltos y conserva los bordes | Imagen con **puntos blancos y negros** (píxeles dañados, interferencia) | Primer paso |
| | Bilateral | Quita el ruido **sin borrar los bordes** | Cuando **los bordes importan** (placas, piezas, texto) | Antes de buscar bordes o leer texto |
| **2. Dar nitidez** | Enfoque | Hace la imagen más nítida | Foto **un poco borrosa** que va a ver una **persona** | **Al final**, solo para mostrar |
| **3. Encontrar bordes** | Sobel | Marca los bordes **verticales** | Cuando importa la **dirección** de las líneas | Después de quitar el ruido |
| | Laplaciano | Marca los bordes **en todas las direcciones** | Imágenes limpias; saber si una foto está enfocada | Después de quitar el ruido |
| | Canny | Dibuja **contornos finos y limpios** | Encontrar **formas y contornos** de objetos | Después de quitar el ruido |
| **4. Luz y blanco/negro** | Mejorar contraste (CLAHE) | Aclara las zonas oscuras sin quemar las claras | **Contraluz**, lugares oscuros, mala iluminación | **Al inicio** |
| | Umbral (blanco y negro) | Separa la imagen en blanco y negro con **un solo corte** | Separar objeto y fondo con **luz pareja** | Para crear la máscara |
| | Umbral adaptativo | Blanco y negro **por zonas** | Separar objeto y fondo con **sombras** o luz desigual (documentos) | Para crear la máscara |

---

#### Grupo 1 — Quitar ruido

**Di (introducción del grupo):**
> "El **ruido** son manchitas, granos o puntos que **no pertenecen a la escena**: los pone la cámara, no el mundo real. Los cuatro primeros filtros sirven para quitarlo. Se usan **al principio**, antes de analizar, porque si no la computadora confunde el ruido con información."

**1. Desenfoque (promedio)**
- **Qué es:** mezcla cada punto de la imagen con sus vecinos, todos por igual. La imagen queda borrosa de forma pareja.
- **Di:**
  > "Es el filtro más simple: cada punto se mezcla con los que lo rodean, y así desaparecen los detalles pequeños, incluido el ruido. El problema es que también se pierden los bordes."
- **Cuándo usarlo:** casi nunca en un proyecto real. Sirve para entender la idea; en la práctica se usa el gaussiano.

**2. Gaussiano**
- **Qué es:** también mezcla cada punto con sus vecinos, pero dando más importancia a los más cercanos. El resultado es un suavizado más natural.
- **Di:**
  > "Es el suavizado **por defecto** de la visión artificial. Quita el **grano** de la imagen, ese ruido fino que aparece en las fotos con poca luz. Es como mirar la imagen con los ojos un poco entrecerrados: los detalles mínimos desaparecen y queda lo importante."
- **Cuándo usarlo:**
  - Imagen **granulada**: cámara de seguridad de noche, cámara barata, poca luz.
  - **Antes** de buscar bordes, antes de umbralizar y antes de detectar movimiento en video (lo veremos en la Sección 5).
- **Cuándo no:** si la imagen tiene **puntos blancos y negros sueltos**: los convierte en manchas en vez de borrarlos.

**3. Mediana**
- **Qué es:** para cada punto, mira a sus vecinos y se queda con el valor "del medio", el más típico. Un punto que es muy distinto de todos los demás simplemente desaparece.
- **Di:**
  > "Imaginen que en una sala de diez personas entra alguien muy alto. Si sacamos el **promedio** de estatura, ese alto lo sube. Si tomamos la estatura **del medio**, la **mediana**, ni se nota. Eso hace este filtro: un punto blanco suelto en medio de una zona oscura es 'el alto de la sala', y la mediana lo ignora. Además, **conserva los bordes** mejor que el gaussiano."
- **Cuándo usarlo:**
  - Imagen con **puntos blancos y negros sueltos** (se llama ruido de **sal y pimienta**): píxeles dañados del sensor, interferencia, cámaras viejas, errores de transmisión.
  - Como **primer paso** de limpieza.
- **Qué mirar en la galería ★ (comparación con el gaussiano):** el gaussiano difumina todo por igual; la mediana deja los **bordes más definidos**, con un aspecto como "pintado".

**4. Bilateral**
- **Qué es:** suaviza solo entre puntos **parecidos**. Si dos puntos vecinos son muy distintos (porque hay un borde entre ellos), no los mezcla.
- **Di:**
  > "El bilateral es el más inteligente del grupo: quita el ruido de las zonas lisas, pero **respeta los bordes**, porque nunca mezcla lo que está a un lado de un borde con lo que está al otro. Es lo que usan las apps de fotos para 'suavizar la piel' sin borrar los ojos ni la boca."
- **Cuándo usarlo:**
  - Cuando hay ruido **y los bordes son lo importante**: leer una **placa**, medir una **pieza**, preparar un texto para leerlo con OCR.
  - Antes de buscar bordes, cuando el gaussiano los deja demasiado borrosos.
- **Cuándo no:** en video en tiempo real o con imágenes muy grandes: es **mucho más lento** que los demás.

---

#### Grupo 2 — Dar nitidez

**5. Enfoque (*sharpen*)**
- **Qué es:** exagera las diferencias entre cada punto y sus vecinos, así que los bordes se marcan más y la imagen se ve más nítida.
- **Di:**
  > "Es lo contrario del suavizado: en vez de mezclar, **resalta** las diferencias. La foto se ve más nítida. Pero ojo: el ruido también es una diferencia, así que el enfoque **también resalta el ruido**. Y no agrega información nueva: solo hace que una persona la vea mejor. A la computadora normalmente no le ayuda."
- **Cuándo usarlo:** foto **un poco borrosa** que va a ver una **persona**: un informe, una pantalla de monitoreo, una evidencia.
- **En qué momento:** **al final** del proceso, y solo para mostrar. **Nunca** antes de analizar.

---

#### Grupo 3 — Encontrar bordes

**Di (introducción del grupo):**
> "Un **borde** es el lugar donde la imagen **cambia de golpe** de claro a oscuro: el contorno de una persona contra la pared, el borde de un casco, el marco de una puerta. Encontrar los bordes es encontrar las **formas**. Estos filtros van **después** de quitar el ruido, porque cada grano de ruido también es un cambio de golpe y aparecería como un borde falso."

**6. Sobel (bordes verticales)**
- **Qué es:** busca los lugares donde la imagen cambia de claro a oscuro **de izquierda a derecha**. Por eso marca las líneas **verticales**.
- **Di:**
  > "Sobel tiene dirección. La versión que ven aquí detecta cambios de izquierda a derecha, así que marca los bordes **verticales**: los costados de las personas, las columnas. Hay otra versión para los bordes horizontales, y juntas dan todos."
- **Cuándo usarlo:** cuando importa la **dirección** de los bordes: líneas de un carril, estructuras verticales u horizontales, texturas con una orientación. Además, es la base de Canny.

**7. Laplaciano (todos los bordes)**
- **Qué es:** marca los bordes **en todas las direcciones a la vez**.
- **Di:**
  > "El Laplaciano no tiene dirección: marca **todos** los bordes. Pero es **muy sensible al ruido**: cualquier puntito distinto de sus vecinos le parece un borde. Un uso curioso: sirve para saber si una foto está **enfocada**. Una foto nítida tiene muchos bordes marcados; una borrosa, casi ninguno."
- **Cuándo usarlo:** sobre imágenes **limpias**, o después de quitar bien el ruido. También para descartar automáticamente fotos movidas o desenfocadas.
- **Qué mirar en la galería ★ (comparación con Sobel):** Sobel solo marca las líneas **verticales**; el Laplaciano marca **todas**, pero con más puntitos de ruido.

**8. Canny (bordes finos)**
- **Qué es:** el detector de bordes más usado. Entrega **líneas blancas finas y continuas** sobre fondo negro: el contorno de las cosas.
- **Di:**
  > "Canny es un detector 'completo': primero suaviza un poco la imagen, después busca los bordes, los **adelgaza** para que queden como una línea fina y al final **decide** cuáles son bordes de verdad y cuáles son ruido. Por eso da un dibujo limpio, como hecho a lápiz, y no manchas grises."
- **Cuándo usarlo:** encontrar **contornos y formas** de objetos, medir piezas, detectar líneas o círculos.
- **En qué momento:** después de quitar el ruido. Lo usamos a fondo en la Sección 3.
- **Qué mirar en la galería:** Sobel y Laplaciano dan grises de distintas intensidades; Canny da una respuesta de **sí o no**: blanco = borde, negro = no borde.

---

#### Grupo 4 — Mejorar la luz y pasar a blanco y negro

**9. Mejorar contraste (CLAHE)**
- **Qué es:** aclara las zonas oscuras y mejora el contraste **por zonas**: cada parte de la imagen se ajusta por separado.
- **Di:**
  > "Cuando una foto tiene una parte muy oscura y otra muy clara, como a contraluz, si la aclaramos toda por igual, la parte clara se 'quema'. CLAHE ajusta **cada zona por separado**: aclara las sombras sin quemar lo que ya estaba bien iluminado. Además tiene un freno para no exagerar."
- **Cuándo usarlo:**
  - Escenas a **contraluz**, almacenes oscuros, cámaras nocturnas, imágenes médicas.
  - Antes de detectar **rostros** con mala iluminación (lo veremos en la Sección 6).
- **En qué momento:** **al inicio**. Pero si la imagen tiene mucho ruido, **primero** se limpia, porque CLAHE también hace más visible el ruido.

**10. Umbral (blanco y negro)**
- **Qué es:** convierte la imagen en **solo blanco y negro**: lo que es más claro que un valor de corte queda blanco; lo demás, negro. Aquí el valor de corte lo elige la computadora automáticamente (método de Otsu).
- **Di:**
  > "El umbral es como decir 'todo lo claro es objeto, todo lo oscuro es fondo'. El resultado se llama **máscara**: una imagen en blanco y negro que separa lo que nos interesa del resto. Es la base para **contar** objetos. El problema: usa **el mismo corte para toda la imagen**."
- **Cuándo usarlo:** separar **objeto y fondo** cuando la **luz es pareja**: piezas sobre una banda con fondo uniforme, texto escaneado, siluetas.
- **En qué momento:** después de quitar el ruido. Su resultado casi siempre se limpia con **morfología** (Sección 4) antes de contar.
- **Cuándo no:** con **sombras** o luz que llega de un solo lado.

**11. Umbral adaptativo**
- **Qué es:** también pasa a blanco y negro, pero decide **zona por zona**: cada punto se compara con lo que tiene **alrededor**, no con toda la imagen.
- **Di:**
  > "El adaptativo no pregunta '¿este punto es oscuro?', sino '¿este punto es **más oscuro que lo que lo rodea**?'. Por eso, aunque media hoja esté en sombra, las letras siguen siendo más oscuras que el papel que las rodea, y las encuentra igual."
- **Cuándo usarlo:** **documentos fotografiados** con el celular (antes de leer el texto con OCR), piezas iluminadas de un lado, cualquier imagen con **sombras**.
- **Qué mirar en la galería ★ (comparación con el umbral normal):** el umbral normal "se come" las zonas en sombra (quedan todas negras o todas blancas); el adaptativo **dibuja los contornos** en toda la imagen.

---

**Di (cierre de la galería):**
> "Resumen en una frase: **quitar el ruido y mejorar la luz al principio, buscar bordes o pasar a blanco y negro en el medio, y dar nitidez solo al final, para mostrar.** Ahora veamos casos reales, donde elegir el filtro equivocado arruina el resultado."

---

### ¿Qué filtro uso y cuándo? 7 casos reales · Compruébalo: elige un caso (4 min, la parte más importante)

**Muestra en pantalla:** la tabla del notebook con los 7 casos (situación real, lo que ves, filtro y momento) y la frase: *"La pregunta no es ¿qué filtro es mejor?, sino ¿qué problema tiene mi imagen y qué quiero hacer después?"*

**Di:**
> "Vamos a ver 7 problemas reales. En cada uno, **antes de mostrar la solución, díganme en el chat qué filtro usarían.**"

**Haz:** para cada caso, **pregunta primero** (que respondan en el chat o con el micrófono) y después selecciónalo en el menú **"Caso:"**. Cada caso muestra tres paneles: el problema, el filtro equivocado (MAL) y el correcto (BIEN).

**Valores esperados y qué decir:**

| Caso | Panel MAL | Panel BIEN | Di |
|---|---|---|---|
| 1. Cámara nocturna | Bordes sin filtrar: puro ruido | Gaussiano y luego Canny: contornos limpios | "Cada grano parece un borde. Hay que quitar el ruido **antes** de buscar bordes." |
| 2. Interferencia | Gaussiano: los puntos quedan como manchas | Mediana: los puntos desaparecen | "Para puntos sueltos, la **mediana**: ignora al 'alto de la sala'." |
| 3. Placa con ruido | Gaussiano fuerte: bordes borrosos | Bilateral: bordes nítidos | "El bilateral limpia sin borrar los bordes de las letras. Es más lento, pero aquí vale la pena." |
| 4. Contraluz | Mejorar contraste a toda la imagen por igual: quema zonas (REGULAR) | CLAHE: mejora por zonas | "CLAHE ajusta cada zona por separado. Va **al inicio**, y también ayuda a detectar rostros." |
| 5. Documento con sombra | Umbral normal: el lado oscuro se pierde | Umbral adaptativo: todo legible | "Con sombras, el **adaptativo**. Es el paso previo típico a leer el texto (OCR)." |
| 6. Contar chalecos | Máscara directa: **17 "objetos"** | Apertura + cierre: **2 objetos** | "Después de pasar a blanco y negro hay que **limpiar la máscara** antes de contar. Eso es la morfología, Sección 4." |
| 7. Foto borrosa | Enfoque con ruido: el ruido se amplifica | Enfoque: más nítida | "El enfoque va **al final** y solo para que la vea una persona. Si hay ruido, lo empeora." |

**Si un alumno pregunta por los tipos de ruido (casos 1 y 2):**

| Ruido | Cómo se ve | De dónde viene | Qué filtro usar |
|---|---|---|---|
| Granulado (gaussiano) | Grano fino en toda la imagen | Poca luz, cámara barata, sensor caliente | Gaussiano; bilateral si importan los bordes |
| Sal y pimienta | Puntos blancos y negros sueltos | Píxeles dañados, interferencia, errores de transmisión | **Mediana** |
| Compresión | Cuadritos y halos alrededor de los bordes | Fotos o videos muy comprimidos (JPEG, WhatsApp) | Filtros especiales; mejor evitar comprimir tanto |

---

### Diseña tu propio kernel (1 min o para casa)

**Di:**
> "Todos los filtros que vimos son, por dentro, una **tablita de 9 números** que se pasa por la imagen, como vimos al inicio de esta sección. En este cuadro pueden escribir sus propios 9 números y ver qué pasa. No hace falta entender la matemática: prueben y miren el resultado."

**Haz:** escribe estos valores en el cuadro de texto, uno por uno.

| Escribe | Qué pasa | Di |
|---|---|---|
| `0,-1,0,-1,5,-1,0,-1,0` (inicial) | La imagen se ve más nítida | "Este es el enfoque de la galería." |
| `1,1,1,1,1,1,1,1,1` + casilla "Promediar" | La imagen se ve borrosa | "Este es el desenfoque: todos los vecinos valen lo mismo." |
| `1,1,1,1,1,1,1,1,1` **sin** "Promediar" | La imagen queda casi blanca | "Sin promediar, la imagen se aclara de más hasta quedar blanca. Por eso hay que promediar." |
| `-1,-1,-1,0,0,0,1,1,1` | Aparecen los bordes horizontales | "Con números negativos y positivos se detectan bordes." |
| `0,1,2,-1,0,1,-2,-1,0` | Aparecen los bordes diagonales (resalta la cinta amarilla y negra) | "Cambiando el orden de los números, cambia la dirección del borde." |
| `0,0,0,0,0,0,0,0,1` | La imagen se desplaza 1 píxel | "Ni suaviza ni enfoca: solo mueve la imagen." |

**Di (la idea que importa):**
> "Con la misma operación y solo cambiando 9 números, obtuvimos suavizado, nitidez, bordes y hasta un desplazamiento. Guarden esta idea: las **redes neuronales**, como YOLO, hacen exactamente esto, pero **aprenden ellas mismas** miles de estas tablitas."

**Si preguntan:**
- **¿Gaussiano o mediana?** Gaussiano para imágenes **granuladas**; mediana para **puntos blancos y negros sueltos**.
- **¿Por qué no usar siempre el bilateral, si es el que mejor respeta los bordes?** Porque es mucho más lento. En video o con muchas imágenes, el gaussiano suele bastar.
- **¿Por qué el enfoque va al final?** Porque también resalta el ruido y no agrega información: solo ayuda a que una persona vea mejor la imagen.
- **¿Qué diferencia hay entre mejorar el contraste a toda la imagen y CLAHE?** Mejorar toda la imagen por igual quema las zonas claras; CLAHE ajusta cada zona por separado.
- **¿Por qué el umbral normal "se come" las sombras?** Porque usa el mismo corte para toda la imagen; en la zona en sombra, todo queda por debajo del corte.
- **¿Por qué la galería está en gris?** Porque la mayoría de estos filtros trabajan sobre el **brillo**, y en gris hay un solo número por punto en vez de tres. Es más simple y más rápido, y la forma de las cosas se ve igual.
- **¿Puedo combinar filtros?** Sí, y es lo normal. Por ejemplo: mejorar contraste → quitar ruido → buscar bordes. Lo importante es el **orden**, que es justo el tema de la siguiente sección.

**Transición:**
> "Ya saben qué hace cada filtro. Pero hay un error muy común que no tiene que ver con elegir el filtro, sino con el **orden** en que se aplican."

---

## 3. El orden importa: ruido → filtro → bordes (min 20–24)

**Di:**
> "El error más común es buscar bordes directamente sobre una imagen con ruido. La regla es: **primero limpiar, después analizar**. En esta sección arman su propio proceso: eligen el ruido, el filtro y los umbrales del detector de bordes."

**Di (qué es Canny, antes de mover nada):**
> "El detector de bordes que vamos a usar es **Canny**, el estándar desde 1986. Hace cuatro pasos:"
>
> "Uno: **suaviza** con un gaussiano pequeño. Dos: calcula la **pendiente con Sobel**, en magnitud y dirección. Tres: **adelgaza** los bordes, quedándose solo con el píxel más fuerte a lo ancho, para que queden de un píxel de grosor. Cuatro: usa **dos umbrales**. Lo que supera el alto es borde seguro; lo que está entre el bajo y el alto solo se acepta si está **conectado** a un borde seguro; lo demás se descarta. Así conserva líneas continuas y descarta puntos sueltos."

**Di (qué es el PSNR):**
> "Para medir qué tan bien limpia cada filtro usamos el **PSNR**: compara la imagen procesada con la original sin ruido y da un número en decibelios. **Más alto es mejor.** Más de 30 dB es buena calidad; menos de 20 dB, muy degradada."

**Haz:** mueve los valores en este orden.

**Valores esperados:**

| Paso | Ruido | Filtro | PSNR | Bordes de Canny | Di |
|---|---|---|---|---|---|
| 1 | Sal y pimienta (30) | **ninguno** (inicial) | 17.6 dB | ~30 % de la imagen son "bordes": puro ruido | "Así se ve buscar bordes sin limpiar." |
| 2 | Sal y pimienta (30) | gaussiano | 27.3 dB | ~16 %: mejora, pero quedan manchas | "Filtro equivocado para este ruido." |
| 3 | Sal y pimienta (30) | **mediana** | **32.8 dB** | **~3.5 %**: limpio | "Filtro correcto." |
| 4 | Gaussiano (30) | ninguno | 18.8 dB | ~37 %: puro ruido | — |
| 5 | Gaussiano (30) | gaussiano | 28.3 dB | ~13 % | "Sirve." |
| 6 | Gaussiano (30) | **bilateral** | **28.7 dB** | **~4 %**: bordes limpios | "El mejor si los bordes importan." |

(Como referencia, la imagen sin ruido tiene ~5 % de bordes.)

**Haz:** muestra otros dos controles.
- **Umbrales de Canny:** bajo 10 / alto 30 da textura del piso por todas partes; bajo 150 / alto 250 hace desaparecer contornos de las personas.
- **"Mejorar contraste" con ruido activado.**

**Di:**
> "El CLAHE también realza el ruido. Por eso, con ruido fuerte, **primero** se limpia y después se mejora el contraste."

**Si preguntan:**
- **¿Siempre hay que suavizar antes de Canny?** Casi siempre. Canny trae un suavizado interno pequeño (5×5), pero con ruido fuerte no alcanza (paso 1: 30 % de "bordes").
- **¿Cómo elijo los umbrales de Canny?** Alto ≈ 2–3 veces el bajo (el notebook usa 50 / 150). Heurística automática: 0.67 y 1.33 veces la mediana de la imagen. Luego ajustar viendo el resultado.
- **¿Un borde es un objeto?** No. Canny entrega líneas, no "el contorno del auto". Para objetos hace falta segmentar.

**Transición:**
> "Hasta ahora hemos trabajado con toda la imagen. Pero muchas veces solo nos importa una parte, y además las máscaras casi nunca salen limpias. De eso trata la siguiente sección."

---

## 4. Regiones de interés, máscaras y morfología (min 24–30)

**Di (ROI y máscaras):**
> "Una **región de interés**, o **ROI**, es la parte de la imagen que nos interesa analizar. Si es un rectángulo, es simplemente un recorte. Si es un polígono, por ejemplo un carril o una zona restringida, se representa con una **máscara**: una imagen en blanco y negro donde **blanco significa sí** y **negro significa no**."
>
> "Limitar el análisis a una ROI ahorra cálculo, evita falsas alarmas fuera de la zona y permite reglas como 'solo me importa esta zona'. Esta zona poligonal la vamos a usar con YOLO en la Sección 8."

**Haz:** muestra el resultado de `pointPolygonTest`.

**Di:**
> "`pointPolygonTest` nos dice si un punto está dentro de la zona: devuelve +1 dentro, −1 fuera y 0 sobre el borde. El punto (700, 450) está dentro y el (200, 300) fuera."

**Muestra en pantalla:** el texto del notebook: *"Las máscaras casi nunca salen limpias. La morfología las arregla"* y la lista de operaciones.

**Di (morfología):**
> "Las máscaras casi nunca salen limpias: tienen puntitos de ruido y objetos partidos, como nuestros chalecos. La **morfología matemática** las arregla cambiando la forma de las manchas blancas. La crearon Georges Matheron y Jean Serra en los años 60, en la Escuela de Minas de París, para analizar minerales en el microscopio."
>
> "Usa un **elemento estructurante**: una pequeña forma, un cuadrado, una cruz o, en nuestro caso, una elipse, que se pasea por la máscara. A diferencia de la convolución, **no multiplica ni suma**: solo pregunta qué hay debajo."

**Di (la erosión, paso a paso):**
> "Empecemos por la operación más simple: la **erosión**. Pongan el elemento, un cuadrado de 3 por 3, encima de cada píxel. El píxel central **sobrevive**, queda blanco, solo si **todo** el cuadrado cae sobre blanco. Si toca aunque sea un píxel negro, se apaga."
>
> "Resultado: a cada objeto se le **come una capa** del borde. Lo que es más delgado que el elemento **desaparece por completo**. El nombre viene de la geología: igual que el mar desgasta la orilla de una isla, las islas pequeñas desaparecen y los istmos delgados se cortan."
>
> "La **dilatación** es lo contrario: el píxel se vuelve blanco si **al menos uno** de los píxeles bajo el elemento es blanco. Es como pintar el contorno con una brocha gorda: los objetos engordan, los huecos pequeños se rellenan y los objetos cercanos se unen."

**Di (apertura y cierre):**
> "Si hacemos primero una erosión y después una dilatación, eso se llama **apertura**: los puntitos de ruido desaparecen con la erosión y ya no vuelven, pero los objetos grandes recuperan su tamaño. Borra el ruido sin achicar los objetos."
>
> "Al revés, primero dilatación y después erosión, se llama **cierre**: la dilatación tapa los huecos y une las partes cercanas, y la erosión devuelve el objeto a su tamaño. Es lo que necesitamos para las franjas reflectivas que parten los chalecos."
>
> "Ojo con el orden: si dilatamos sin limpiar antes, los puntitos de ruido se pegan al objeto y ya no se pueden separar. Por eso en el video vamos a hacer primero la apertura y después el cierre."

**Resumen de las operaciones:**

| Operación | Regla | Efecto |
|---|---|---|
| **Erosión** | Blanco solo si **todos** bajo el elemento son blancos | Adelgaza; borra lo más pequeño que el elemento; corta uniones delgadas |
| **Dilatación** | Blanco si **al menos uno** es blanco | Engorda; rellena huecos pequeños; une objetos cercanos |
| **Apertura** | Erosión y luego dilatación | Borra lo pequeño; lo grande recupera su tamaño |
| **Cierre** | Dilatación y luego erosión | Rellena huecos y une partes cercanas sin engordar |
| **Gradiente** | Dilatación − erosión | Deja solo el contorno |

**Muestra en pantalla:** la pregunta del notebook: *"¿Qué operación une mejor cada chaleco en una sola mancha?"*

**Haz:** en el control **"Operación" / "Tamaño"**, recorre las operaciones. La máscara de los chalecos empieza con **17 manchas** (lo ideal son 2).

**Valores esperados:**

| Operación | Tamaño 5 | Tamaño 9 (inicial) | Tamaño 15 | Tamaño 31 | Di |
|---|---|---|---|---|---|
| Cierre | 14 | 10 | **2** | 2 | "El cierre **une** las partes de cada chaleco: con 15 ya son 2." |
| Dilatación | 10 | 6 | **2** | 2 | "También une, pero **engorda** el objeto." |
| Apertura | 8 | 4 | 4 | 1 | "Borra puntitos; con tamaño grande borra objetos completos." |
| Erosión | 8 | 4 | 4 | 1 | "Adelgaza hasta hacer desaparecer." |
| Gradiente | — | — | — | — | "Deja solo el borde de cada mancha." |

**Di (por qué con 15 ya funciona):**
> "El cierre une partes separadas por huecos más angostos que el elemento. Las franjas reflectivas que parten cada chaleco son más anchas de lo que cubre un elemento de 9, pero no de uno de 15. Y con la apertura de 31 queda **1**: borró un chaleco entero. El tamaño del elemento se elige según el tamaño de lo que quieren borrar o unir."

**Pregunta al grupo:** "¿Cierre o dilatación para unir?" (Respuesta: cierre, porque une sin cambiar tanto el tamaño final.)

**Si preguntan:**
- **¿Qué diferencia hay entre erosión y apertura?** La erosión achica todo; la apertura borra lo pequeño pero devuelve a los objetos grandes su tamaño.
- **¿Por qué la máscara de los chalecos sale en pedazos?** Las franjas reflectivas no son naranjas. Por eso existe la morfología.
- **¿Por qué la erosión engordó mi objeto?** Porque el objeto es negro sobre fondo blanco: la morfología de OpenCV actúa sobre el **blanco**. Hay que invertir la máscara (`THRESH_BINARY_INV` o `cv2.bitwise_not`).
- **¿Cómo elijo el tamaño del elemento estructurante?** Según el tamaño de lo que se quiere borrar (apertura) o de los huecos que se quieren cerrar (cierre). Ni más pequeño (no alcanza) ni mucho más grande (borra o fusiona objetos reales).
- **¿Qué diferencia hay entre un kernel y una máscara?** El kernel es una tabla de **pesos** que se desliza; la máscara es una imagen binaria que dice **dónde** mirar.
- **¿Puedo dibujar la ROI con el mouse?** Fuera de Colab sí: `cv2.selectROI(imagen)` para un rectángulo, o clics de Matplotlib para los vértices de un polígono.

**Transición:**
> "Un video es una secuencia de imágenes. Así que todo lo que vimos se aplica cuadro a cuadro. Vamos a construir un contador de autos **sin inteligencia artificial**, solo con filtros, y lo vamos a correr dos veces: sin filtros y con filtros."

---

## 5. Video: contar vehículos, sin filtros y con filtros (min 30–38)

**Di (sustracción de fondo):**
> "Si la cámara está quieta, todo lo que **cambia** respecto al fondo es algo que se mueve. Esa es la idea de la **sustracción de fondo**."
>
> "La versión más simple sería restar un cuadro del anterior. Nosotros usamos **MOG2**, que aprende cómo se ve la calle vacía. Modela **cada píxel** con una mezcla de varias campanas de Gauss, porque un píxel de fondo puede tener varios estados normales: una hoja que se mueve, el reflejo del agua. Usa los últimos **200 cuadros** para aprender y se va actualizando, así se adapta a cambios lentos de luz."
>
> "Y tiene un truco: si un píxel es **más oscuro que el fondo pero del mismo color**, lo marca como **sombra**, en gris (valor 127), en vez de blanco (255)."

**Haz:** muestra la tabla de 7 pasos del notebook.

**Di (los filtros en orden):**
> "Miren dónde va cada filtro. Uno: **suavizado gaussiano antes** de comparar con el fondo, para quitar el grano de la cámara. Dos: sustracción de fondo. Tres: **umbral** en 200, que conserva el blanco y elimina el gris de las sombras. Cuatro: **apertura**, para borrar puntitos. Cinco: **cierre y dilatación**, para unir las partes de cada auto. Seis: **contornos**, cada mancha grande es un vehículo. Siete: **seguimiento** y línea de conteo."
>
> "¿Por qué primero la apertura y después el cierre? Por lo que vimos en la Sección 4: si unimos primero, los puntitos de ruido se pegan al auto y ya no se pueden borrar."

**Di (seguimiento por centroides):**
> "Para contar no basta con detectar: hay que saber que el auto de este cuadro es **el mismo** del cuadro anterior. Eso se llama **seguimiento** o *tracking*. Nuestro tracker es el más simple: calcula el **centroide**, el centro de cada mancha, y lo une con el centroide más cercano del cuadro anterior, si está a menos de 60 píxeles. Si no hay ninguno cerca, le da un **ID nuevo**. Cuando un ID cruza la línea roja, se cuenta una vez."

**Haz:** ejecuta el video lado a lado.

**Valores esperados:**

| | Autos contados (reales: 4) | IDs creados |
|---|---|---|
| Sin filtros | 4 | **40** |
| Con filtros | 4 | **10** |

**Di (mensaje clave de la sección):**
> "Sin filtros también cuenta 4. ¿Entonces funciona? **No.** Miren la máscara: es ruido y sombras, y el sistema está persiguiendo 40 objetos falsos. Acertó **por casualidad**. Con filtros cada auto es una sola mancha: acierta **por las razones correctas**."
>
> "En un sistema perfecto, el número de IDs sería igual al de objetos reales. Cada ID de más es una mancha falsa o un objeto que se partió o se perdió. Por eso hay que mirar **la máscara y los IDs**, no solo el número final."

### Activa y desactiva cada filtro

**Haz:** desactiva las casillas **una a la vez**. Avisa que cada cambio procesa el video completo y tarda unos segundos.

**Valores esperados:**

| Filtro desactivado | Contados | IDs | Di |
|---|---|---|---|
| Ninguno (todos activos) | 4 | 10 | "Referencia." |
| Sin suavizado | 5 | 14 | "El grano de la cámara genera manchas falsas." |
| **Sin quitar sombras** | 3 | **67** | "**El filtro más importante:** las sombras se vuelven objetos." |
| Sin apertura | 3 | 21 | "El ruido se cuela en la máscara." |
| Sin cierre + dilatación | **6** | 24 | "Cada auto se parte en varias manchas y **se cuenta dos veces**." |

**Di (por qué las sombras son el peor problema):**
> "Una sombra es un cambio real en la imagen y se mueve con el auto: para la sustracción de fondo **es movimiento**. MOG2 la marca en gris y el umbral en 200 la elimina. Si desactivamos ese umbral, pasa a 0 y el gris se vuelve objeto: 67 IDs."

**Pregunta al grupo:** "Si sin filtros cuenta 4, ¿el sistema funciona?" (Respuesta: no; acertó por casualidad.)

**Di (cierre de la sección):**
> "Ventajas de este enfoque: es rápido, no necesita datos de entrenamiento y es fácil de entender. Funciona bien con **cámara fija** y escenas controladas. Pero aun con filtros quedan IDs de más: la cámara cambia de brillo y el sistema **no sabe qué es** cada mancha. Una persona, un perro o una bolsa al viento contarían igual."

**Si preguntan:**
- **¿Por qué la sustracción de fondo no funciona con la cámara en movimiento?** Porque todo cambia de un cuadro a otro, y todo parece "objeto". En drones o cámaras que giran se necesita otro enfoque.
- **¿Qué pasa si un auto se detiene?** Tras un tiempo (según `history`) el modelo lo integra al fondo y deja de detectarlo; cuando se va, deja un "fantasma".
- **¿Y si pasa una nube o la cámara ajusta su exposición?** Cambia toda la imagen de golpe y aparecen falsas detecciones por todas partes.
- **¿Por qué hay detecciones raras al principio del video?** El modelo todavía no ha aprendido el fondo.
- **¿Por qué importan los IDs si el conteo da bien?** Porque revelan si el sistema funciona por las razones correctas. 40 IDs para 4 autos significa que el acierto fue casualidad.

**Transición:**
> "Hasta ahora detectamos manchas que se mueven, sin saber qué son. Pasemos a detectar algo concreto, que además es el primer paso del proyecto final: rostros."

---

# Parte 2 — Visión artificial: rostros y YOLO26

**Muestra en pantalla:** el encabezado **"Parte 2 — Visión artificial: rostros y YOLO26"**.

---

## 6. Detección de rostros con OpenCV (Viola-Jones) (min 38–43)

**Di (anuncio del proyecto final):**
> "El proyecto final es un sistema de reconocimiento facial. Y antes de reconocer a alguien, hay que **encontrar su rostro**. Si ese primer paso falla, todo lo demás falla: cada rostro que no se detecta es una persona que nunca será identificada."

**Di (cómo funciona Viola-Jones):**
> "OpenCV trae un detector clásico de 2001, de Paul Viola y Michael Jones, que estuvo durante años dentro de las cámaras digitales. Se basa en cuatro ideas."
>
> "**Uno, características tipo Haar:** rectángulos blancos y negros pegados. Se suma lo que hay bajo el blanco y se resta lo que hay bajo el negro. Por ejemplo, la franja de los ojos es más oscura que la de las mejillas, y el puente de la nariz es más claro que los ojos. Son kernels muy simples, como los de la Sección 2."
>
> "**Dos, la imagen integral:** una tabla precalculada que permite sumar los píxeles de **cualquier** rectángulo con solo 4 lecturas, sin importar su tamaño. Eso lo hizo rapidísimo."
>
> "**Tres, AdaBoost:** en una ventana de 24 por 24 píxeles hay más de 160 000 rectángulos posibles. Un algoritmo de aprendizaje automático, AdaBoost, elige los más útiles y combina muchos clasificadores débiles en uno fuerte."
>
> "**Cuatro, la cascada:** los clasificadores se organizan en etapas. La primera usa muy pocas características y descarta enseguida casi todas las ventanas que obviamente no son una cara. Solo las candidatas pasan a las etapas más costosas."
>
> "Así detectaba caras en tiempo real en una computadora de 2001. Pero tiene un límite: se entrenó con rostros **de frente**."

**Haz:** explica los deslizadores antes de moverlos.

**Di:**
> "La ventana de búsqueda tiene un tamaño fijo. Para encontrar caras de distintos tamaños, el detector **achica la imagen** una y otra vez. El **paso entre escalas** es cuánto la achica cada vez: 1.1 es un 10 %, muchas escalas, lento pero preciso; 1.3 es rápido pero se salta tamaños."
>
> "Una cara real se detecta varias veces en ventanas que se solapan; una falsa alarma suele aparecer una sola vez. **Vecinos mínimos** es cuántas detecciones agrupadas exigimos para aceptar una cara."
>
> "**Tamaño mínimo** ignora las caras más pequeñas que ese valor en píxeles: ahorra tiempo y falsas alarmas, pero puede perder caras lejanas."

**Haz:** mueve los deslizadores en este orden.

**Valores esperados:**

| Frame | Ajuste | Rostros | Di |
|---|---|---|---|
| 488 (inicial) | Valores iniciales | **3 de 3** | "Funciona con rostros frontales y cercanos." |
| 61 | Valores iniciales | 1 de 1 | — |
| **244** | Valores iniciales (tamaño mínimo 30) | **0 de 2** | "Los rostros están lejos: son más pequeños que 30 px." |
| **244** | **Tamaño mínimo = 20** | **2 de 2** | "Era un problema de **parámetro**." |
| 244 | Vecinos mínimos = 1 | 3 (uno es falso) | "Pocos vecinos = falsas alarmas." |
| 488 | Paso entre escalas = 1.3 | 2 de 3 | "Paso grande = más rápido, pero se salta tamaños." |
| 488 | Vecinos mínimos = 10 | 2 de 3 | "Muy exigente = pierde rostros." |
| **305** | Cualquier ajuste razonable | **0 de 1** | "Rostro de perfil y movido: es una **limitación del método**, no de los parámetros." |
| Cualquiera | "Ecualizar contraste antes" | Sin cambio en este video | "Ayuda en escenas oscuras o a contraluz (caso 4 de la Sección 2)." |

**Di (cierre de la sección):**
> "En el frame 244 bastó ajustar un parámetro. En el 305, el método no da para más. En el Taller 2 usaremos un detector moderno que sí encuentra ese rostro."

**Di (detectar no es reconocer):**
> "Ojo con una confusión muy común: hoy solo **detectamos** rostros, es decir, respondemos '¿hay una cara y dónde?'. **Reconocer** es responder '¿de quién es esta cara?'. El proyecto final necesita las dos cosas, en ese orden."

| Tarea | Pregunta | Ejemplo |
|---|---|---|
| **Detección de rostros** | ¿Hay rostros y dónde? | Viola-Jones, YOLO-face |
| **Verificación (1:1)** | ¿Esta persona es quien dice ser? | Desbloquear el celular |
| **Identificación (1:N)** | ¿Quién de mi base de datos es? | Control de acceso |

**Si preguntan:**
- **¿Por qué Viola-Jones no detecta el rostro de perfil?** El modelo que usamos se entrenó con rostros frontales. OpenCV trae otro para perfiles, pero los detectores modernos lo resuelven mejor. También falla con caras inclinadas, mascarillas, lentes oscuros y mala iluminación.
- **¿Viola-Jones es inteligencia artificial?** Sí, es aprendizaje automático (AdaBoost elige las características), pero no aprendizaje profundo: las características son rectángulos fijos, no kernels aprendidos.
- **¿Qué es el archivo `.xml` de cascada?** El clasificador ya entrenado: qué características Haar usar, con qué umbrales y en qué etapas.
- **¿Por qué la ecualización no cambió nada en este video?** Porque el video ya tiene buena iluminación. Ayuda en escenas oscuras o a contraluz.
- **¿Es legal usar reconocimiento facial?** Depende del país y del uso. Los rostros son **datos biométricos**, protegidos como datos sensibles por la mayoría de las leyes de protección de datos. Hay sesgos documentados: el estudio *Gender Shades* (Buolamwini y Gebru, 2018) mostró errores mucho mayores en mujeres de piel oscura en sistemas comerciales. Para el proyecto: consentimiento de las personas, datos mínimos y no publicar rostros.

**Transición:**
> "Todos los filtros y detectores que hemos usado hasta ahora los diseñó una persona. Ahora vamos a ver qué pasa cuando la máquina aprende sus propios filtros."

---

## 7. Tareas de visión artificial con YOLO26 (min 43–51)

### Del filtro diseñado al filtro aprendido

**Muestra en pantalla:** la imagen de YOLO26 que abre la sección y el texto del notebook: *"Los filtros de las secciones anteriores los diseñamos nosotros. Una red neuronal convolucional aprende sus propios filtros a partir de miles de imágenes."*

**Di:**
> "Todo lo que hicimos hasta ahora usa kernels que **diseñó una persona**: el gaussiano, Sobel, los rectángulos de Viola-Jones. Una **red neuronal convolucional**, o CNN, es una pila de capas que hacen exactamente la misma operación de la Sección 2, la convolución, pero sus kernels **no los diseña nadie: se aprenden** a partir de miles de imágenes con su respuesta correcta."
>
> "Y pasa algo muy bonito: cuando se entrena una CNN con millones de fotos, sin decirle nada, sus primeros kernels terminan pareciéndose a los detectores de bordes que vimos en la galería. La red **redescubre** lo que nosotros diseñamos a mano y después sigue aprendiendo cosas que no sabríamos diseñar."

**Di (cómo está organizada una CNN):**
> "Las primeras capas detectan **bordes y colores**. Las intermedias combinan esos bordes en **texturas y partes**: una rueda, un ojo. Las últimas combinan las partes en **objetos completos**: un auto, una cara. ¿Recuerdan el GIF que saltaba de 2 en 2? Así cada capa reduce la imagen y ve una porción más grande de la escena."

**Si preguntan:**
- **¿Por qué se necesita GPU?** Una CNN hace millones de multiplicaciones independientes, y una GPU tiene miles de núcleos que las hacen en paralelo. Los modelos *nano* de YOLO son lo bastante pequeños para correr en CPU.

### YOLO: mirar una sola vez

**Di (de dónde viene YOLO):**
> "Antes de YOLO, los detectores trabajaban en **dos etapas**: primero proponían regiones donde podría haber un objeto y luego clasificaban cada una. Eran precisos pero lentos."
>
> "En 2016, Joseph Redmon y su equipo propusieron **YOLO**: dividir la imagen en una cuadrícula y que cada celda prediga directamente cajas, confianza y clase. Una sola red, **una sola pasada**. Por eso es tan rápido."
>
> "Desde entonces ha habido muchas versiones, de distintos autores. **YOLO26** es la más reciente de Ultralytics, de enero de 2026, y está pensada para funcionar rápido **incluso sin GPU**."

**Di (cómo está construido un detector moderno):**
> "Tiene tres partes. El **backbone**, una CNN que extrae características. El **neck**, o cuello, que combina características de distintos tamaños para ver objetos grandes y pequeños. Y el **head**, o cabeza, que predice para cada posición la caja, la confianza y la clase."

### Las tareas: qué pregunta responde cada una

**Muestra en pantalla:** la tabla de tareas del notebook (Tarea · Pregunta que responde · Modelo) y, debajo, las figuras con los resultados.

**Di:**
> "Cada tarea responde a una pregunta distinta, de menos a más detalle."
>
> "**Clasificación:** ¿qué es esto? Una sola etiqueta para toda la imagen."
>
> "**Detección:** ¿qué hay y dónde? Una caja por cada objeto."
>
> "**OBB**, cajas orientadas: ¿dónde está y con qué ángulo? Cajas que giran con el objeto, muy usadas en imágenes satelitales."
>
> "**Segmentación:** ¿qué píxeles exactos ocupa cada objeto? La silueta completa."
>
> "**Pose:** ¿cómo está el cuerpo? Los puntos de las articulaciones."
>
> "La regla para elegir es usar **la tarea más simple que responda la pregunta del problema**, porque más detalle cuesta más etiquetar, entrenar y procesar."

**Haz:** recorre la tabla de tareas mientras se muestran los resultados.

**Valores esperados:**

| Tarea | Qué es | Cuándo usarla (ejemplo) | Modelo | Aprendió con | Resultado |
|---|---|---|---|---|---|
| Detección | **YOLO Detect** es la tarea base de YOLO (*You Only Look Once*): una red neuronal que, en una sola pasada, localiza **cada objeto** de una imagen o video, lo encierra en una **caja delimitadora** (*bounding box*) e indica su **clase** y la **confianza** de la predicción | Contar o ubicar objetos: personas en una zona, casco sí/no, rostros antes de reconocerlos | `yolo26n.pt` | COCO (80 clases) | 2 personas con caja |
| Segmentación | **YOLO Seg** (segmentación de instancias) es una variante de YOLO que, además de la caja y la clase, genera una **máscara píxel a píxel** de cada objeto: su silueta exacta, separando cada objeto de los demás aunque sean de la misma clase | Cuando importa la forma o el área: medir una herida o grieta, quitar el fondo | `yolo26n-seg.pt` | COCO | 2 personas pintadas píxel a píxel |
| Pose | **YOLO Pose** (estimación de pose) es una variante de YOLO que detecta cada persona y localiza sus **puntos clave** (*keypoints*): 17 articulaciones como hombros, codos, rodillas y tobillos que, unidas, forman un **esqueleto** que describe la postura del cuerpo | Cuando importa la postura: detectar caídas, ergonomía, contar repeticiones | `yolo26n-pose.pt` | COCO (17 puntos del cuerpo) | Esqueleto de cada persona |
| Clasificación | **YOLO Classify** es una variante de YOLO que asigna **una sola etiqueta a toda la imagen** con su probabilidad (por ejemplo, "limón 51 %"). Responde **qué** hay, pero no **dónde** está ni **cuántos** hay | Un objeto por imagen: fruta madura/verde, pieza defectuosa, expresión de un rostro recortado | `yolo26n-cls.pt` | ImageNet (1000 clases) | Limón: **lemon 51 %**, 2.º *spaghetti squash* 21 % |
| OBB | **YOLO OBB** (*Oriented Bounding Box*) es una variante de YOLO que detecta objetos en imágenes o videos identificando su **posición, tamaño y orientación** (ángulo de rotación). Usa **cajas rotadas** que se ajustan al objeto, en vez de rectángulos alineados con la imagen | Objetos girados y juntos vistos desde arriba: barcos o autos en imagen satelital, piezas en una banda | `yolo26n-obb.pt` | DOTA (imágenes aéreas) | Unos **173 objetos rotados** en la foto del puerto |

**Di (los datos con que aprendió cada modelo):**
> "Un modelo **solo conoce las clases de su dataset**. Detección, segmentación y pose aprendieron con **COCO**, de Microsoft: unas 118 000 imágenes con 80 tipos de objetos cotidianos, personas, autos, animales. El clasificador aprendió con **ImageNet**: más de un millón de imágenes en 1000 categorías. Y el de OBB con **DOTA**, imágenes aéreas de aviones, barcos y vehículos. Por eso YOLO de COCO no sabe qué es un casco: para eso hay que entrenarlo con imágenes propias, que es el Taller 2."

**Di (sobre el limón y la confianza):**
> "Miren el limón: 51 %, y la segunda opción es una calabaza. La **confianza no es la probabilidad de acertar**. El clasificador siempre reparte el 100 % entre las 1000 clases que conoce, aunque la imagen no sea ninguna de ellas. Un modelo puede estar 90 % seguro y equivocarse."

**Pregunta al grupo:**
- "¿Qué tarea necesitaría tu proyecto, y por qué no una más simple?"
- "¿Por qué el clasificador duda con el limón?" (Respuesta: la imagen está movida y el fondo no se parece a las fotos con que aprendió; la confianza no garantiza acierto.)

**Detalles del código (si los señalan):**
- `conf=0.4`: solo se dibujan predicciones con confianza ≥ 40 %. Bajarlo muestra más objetos (y más falsos); subirlo, menos.
- La `n` de `yolo26n` es *nano*, el tamaño más pequeño (n, s, m, l, x). Más grande = más preciso y más lento.
- `.pt` es un archivo de pesos de PyTorch: contiene los valores aprendidos de todos los kernels de la red.
- Los modelos se descargan solos la primera vez.

**Si preguntan:**
- **¿YOLO funciona sin GPU?** Sí, la versión nano está pensada para CPU.
- **¿Qué diferencia hay entre YOLO26 y versiones anteriores?** Inferencia sin NMS, cabeza simplificada (sin DFL), mejoras para objetos pequeños, nuevo optimizador y foco en CPU y dispositivos de borde.
- **¿Qué es NMS?** Supresión de no máximos: elimina cajas duplicadas del mismo objeto quedándose con la de mayor confianza. YOLO26 ya no lo necesita.
- **¿51 % significa que acierta la mitad de las veces?** No. Es la salida del modelo, no una probabilidad calibrada de acierto.
- **¿YOLO detecta cascos y chalecos?** No con el modelo de COCO: esas clases no están entre sus 80. Hay que entrenarlo con imágenes propias (Taller 2).
- **¿Puedo usar YOLO con mis propias clases?** Sí, ese es el Taller 2 (*fine-tuning*).
- **¿YOLO26 es libre?** Sí, con licencia AGPL-3.0 (si se ofrece como servicio, hay que publicar el código). Para uso comercial cerrado se necesita la licencia Enterprise de Ultralytics.

**Transición:**
> "Ahora unamos todo lo del día en una aplicación real: una zona de seguridad que avisa cuando una persona entra."

---

## 8. Aplicación: zona de seguridad con detección + seguimiento (min 51–56)

**Di:**
> "Aquí combinamos tres cosas que ya vimos: la **zona poligonal** de la Sección 4, la **detección de personas** de la Sección 7 y el **seguimiento**, que es la versión moderna del tracker de centroides de la Sección 5."

**Di (el punto de los pies):**
> "Para saber si una persona **pisa** la zona usamos el **punto de los pies**, el centro inferior de la caja, no el centro. ¿Por qué? La persona está de pie sobre el piso, y la zona está dibujada en el piso. El único punto de la caja que realmente toca el piso es el de abajo. Por la perspectiva, el centro de la caja cae a la altura del torso, que en la imagen puede quedar 'encima' de otra parte del piso."

**Di (el tracker moderno):**
> "El tracker de la Sección 5 unía cada mancha con la más cercana. Los trackers modernos además **predicen**: si una persona iba caminando a la derecha, en el siguiente cuadro la buscan un poco más a la derecha, no donde estaba. Eso lo hace un **filtro de Kalman**. Ultralytics usa por defecto **BoT-SORT**, que además compensa el movimiento de la cámara."

**Haz:** ejecuta la celda.

**Valores esperados:** 2 personas ingresan a la zona en 30 s; el sistema reporta 2 IDs y muestra la alerta en rojo.

**Di (comparación con la Sección 5):**
> "Comparen con el contador de autos: YOLO **sabe que son personas**, no se confunde con sombras ni con cambios de brillo, y mantiene el ID de cada una."

**Di (el "Experimenta" del notebook, déjalo para casa):**
> "Al final de la sección, el notebook les propone un experimento: cambiar el tracker a ByteTrack. Háganlo en casa y cuenten cuántos IDs aparecen para las mismas 2 personas."

**Detalles del código (si los señalan):**
- `classes=[0]` limita la detección a la clase 0 de COCO (persona).
- `persist=True` le dice al tracker que conserve la memoria entre cuadros; sin eso, cada cuadro empezaría con IDs nuevos.

**Si preguntan:**
- **¿Por qué ByteTrack da más IDs aquí (6 en las pruebas)?** Cada tracker tiene parámetros distintos para crear y mantener IDs; las personas que entran cortadas por el borde reciben IDs nuevos al reaparecer completas.
- **¿Qué es un filtro de Kalman?** Un método que predice la próxima posición de un objeto según su movimiento previo y corrige esa predicción con la nueva medición.
- **Si una persona sale de la imagen y vuelve, ¿mantiene su ID?** Normalmente no: recibe un ID nuevo. Para recuperarlo hace falta reidentificación por apariencia.
- **¿Entonces OpenCV ya no sirve?** Sí sirve: lee el video, prepara las imágenes, define zonas, dibuja resultados y resuelve problemas simples sin entrenar nada. En esta sección OpenCV y YOLO trabajan juntos.

**Transición:**
> "Cerremos con lo que se llevan hoy."

---

## Cierre (min 56–60)

### Lo que te llevas

**Muestra en pantalla:** la celda **"Cierre"** del notebook, con las 4 ideas de "Lo que te llevas".

**Di:**
> "Cuatro ideas para llevarse."
>
> "**Uno:** cada filtro tiene su momento: mejorar contraste, limpiar ruido, analizar con bordes o umbral, y limpiar la máscara con morfología."
>
> "**Dos:** en video, ese mismo orden permite detectar movimiento, seguir y contar objetos sin entrenar nada. Y un sistema puede acertar por casualidad: miren el proceso, no solo el resultado."
>
> "**Tres:** OpenCV detecta rostros frontales, pero pierde los pequeños y los de perfil."
>
> "**Cuatro:** YOLO26 resuelve muchas tareas, detectar, segmentar, pose, clasificar y objetos rotados, con modelos ya entrenados. Pero solo conoce las clases con que aprendió."

### Próximo taller: entrenar y evaluar YOLO26

**Di:**
> "En el Taller 2 abrimos YOLO26 por dentro: cómo medimos si detecta bien, cómo lo entrenamos con nuestras propias imágenes para clases que no conoce, y el pipeline de detección y reconocimiento facial del proyecto final."

### Reto

**Muestra en pantalla:** la lista del **Reto** del notebook y léela con ellos.

1. En la Sección 1, encontrar los valores que detectan **solo los cascos amarillos** (solución: tono 20–35, saturación ≥ 100, brillo ≥ 150).
2. En la Sección 3, encontrar la mejor combinación de filtro y umbrales para cada tipo de ruido.
3. **(Proyecto final)** En la Sección 6, anotar en qué frames OpenCV pierde rostros. Compararemos esa lista con YOLO26 en el Taller 2.
4. Cambiar `classes=[0]` en la Sección 8 para vigilar otra clase, y probar con un video propio.
5. Crear una cuenta en Roboflow para el Taller 2.

**Di (despedida):**
> "El anexo del final es para casa: muestra cómo medir distancias en el piso y encontrar objetos aunque giren. Traigan su cuenta de Roboflow al próximo taller."

---

## Anexo para casa: análisis geométrico

**Di (si lo presentas):**
> "Este anexo es para casa. Muestra cómo OpenCV mide y encuentra cosas con geometría, sin redes neuronales."

### A. Transformaciones: rotar, escalar, inclinar

**Di:**
> "Rotar, escalar, mover e inclinar se representan con una matriz de 2 por 3. Son **transformaciones afines**: conservan las líneas paralelas. Un rectángulo puede volverse un paralelogramo, pero nunca un trapecio."

**Valores sugeridos:** ángulo 90 con escala 0.5 (rotación pura); inclinación 0.3 (*shear*); "Espejo".

### B. Vista cenital y medición sobre el piso

**Di:**
> "Las líneas rojas del piso forman cuadrados, pero la cámara los ve como trapecios. Marcando las 4 esquinas de un cuadrado, OpenCV calcula una **homografía**: una matriz de 3 por 3 que relaciona dos vistas de un **mismo plano**. Con ella se endereza el piso, como si lo viéramos desde un dron. La homografía conserva las líneas rectas, pero no el paralelismo: por eso los rieles del tren se juntan en el horizonte."
>
> "Las columnas se deforman porque **no están en el plano del piso**: la transformación solo vale para ese plano. Así funcionan los sistemas que miden distancias o velocidades con una sola cámara."

**Valores esperados:** solo mirar el GIF. Las líneas del piso quedan en ángulo recto y las columnas se deforman. Los dos trabajadores quedan a 0.50 celdas de distancia.

### C. Encontrar un objeto aunque gire o cambie de tamaño

**Di:**
> "La **coincidencia de plantillas** desliza la foto del objeto sobre la escena y mide cuánto se parece en cada posición, de −1 a 1. Es simple y rápida, pero compara píxel a píxel en la misma orientación: si el objeto gira o cambia de tamaño, falla."
>
> "**ORB** busca **puntos clave**: esquinas distintivas, cada una con una 'huella' binaria que no cambia al girar. Empareja las huellas de la plantilla con las de la escena. Es una alternativa libre a SIFT."

**Valores esperados** ("Coincidencia" de la plantilla, 1 = perfecta, frente a los puntos que coinciden con ORB):

| Giro | Tamaño | Plantilla | ORB | Di |
|---|---|---|---|---|
| 0° | 1.0 | 1.00 (perfecta) | 894 puntos | "Ambos aciertan." |
| 10° | 1.0 | 0.76 | 733 | "La plantilla empieza a fallar con apenas 10°." |
| 25° | 0.7 (inicial) | 0.49: marca el lugar equivocado | 499: la encuentra | "Diferencia clara." |
| 90° | 0.4 | 0.42 | 132: aún la encuentra | "ORB resiste giro y tamaño." |

**Si preguntan:**
- **¿Qué es una homografía?** Una transformación que relaciona dos vistas de un mismo plano; con 4 puntos se puede enderezar el piso y medir sobre él.
- **¿Se puede medir la velocidad de los autos?** Sí: con una homografía del piso se convierten píxeles a metros, y con los fps del video, cuadros a segundos.
- **¿Por qué falla la plantilla al girar el objeto?** Compara píxel a píxel en la misma orientación y escala. ORB describe puntos clave de forma que no cambia con el giro.

---

# APÉNDICES (consulta rápida)

## A. Si algo falla: riesgos técnicos y plan B

| Riesgo | Plan B |
|---|---|
| `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'` | Ocurre con OpenCV 5 si alguien ejecutó celdas antes de instalar. La celda de instalación reinicia la sesión sola una vez. Si persiste: **Entorno de ejecución → Reiniciar sesión** y ejecutar todo de nuevo. |
| La sesión se reinicia al instalar | Es esperado: volver a pulsar **Ejecutar todas**. |
| Los controles no responden | Ejecutar de nuevo la celda. Se actualizan al **soltar** el deslizador. |
| Las casillas del conteo tardan | Es normal: cada cambio procesa el video completo (unos segundos). |
| Los GIF de convolución no cargan (dependen de GitHub) | Abrir en otra pestaña el repositorio conv_arithmetic (enlace en el Material de apoyo) y compartir esa pestaña. |
| GitHub lento con los videos | Tener la carpeta `material/` en tu Drive y copiarla; el código no descarga lo que ya existe. |
| La zona de seguridad tarda en CPU | Bajar `while n < 1800` a `900` (15 s de video). |
| Colab pisa cambios del notebook | Si editas el archivo fuera de Colab, cierra la pestaña de Colab sin guardar y vuelve a abrirlo desde Drive. |

## B. Conceptos que se confunden

| Par de conceptos | Diferencia en una línea |
|---|---|
| Procesamiento de imágenes vs visión artificial | Imagen → imagen mejorada vs imagen → información |
| Kernel vs máscara | El kernel es una tabla de **pesos** que se desliza; la máscara es una imagen binaria que dice **dónde** mirar |
| Convolución vs correlación | La convolución gira el kernel 180°; OpenCV y las CNN usan correlación (igual para kernels simétricos) |
| Gaussiano vs mediana | Promedio ponderado (lineal) vs valor central ordenado (no lineal) |
| Sobel vs Canny | Sobel da la "fuerza" del borde en cada píxel; Canny entrega bordes finos y conectados (usa Sobel por dentro) |
| Ecualización vs CLAHE | Una regla global vs reglas por zonas con límite de contraste |
| Umbral global vs Otsu vs adaptativo | T elegido a mano vs T automático para toda la imagen vs un T por cada vecindario |
| Erosión vs apertura | Achicar todo vs borrar lo pequeño y conservar el tamaño de lo grande |
| Dilatación vs cierre | Engordar todo vs rellenar huecos conservando el tamaño |
| ROI vs máscara | La ROI es la región; la máscara es una forma de representarla |
| Detección vs seguimiento | Encontrar objetos en un cuadro vs mantener su identidad entre cuadros |
| Detección de rostros vs reconocimiento facial | ¿Hay una cara y dónde? vs ¿de quién es? |
| Clasificación vs detección | Una etiqueta para toda la imagen vs etiqueta + caja para cada objeto |
| Segmentación semántica vs de instancias | Píxeles por clase vs píxeles por objeto individual |
| Caja normal vs OBB | Rectángulo alineado con los ejes vs rectángulo rotado |
| Entrenamiento vs inferencia | Ajustar los pesos vs usar los pesos para predecir |
| Confianza vs exactitud | Lo seguro que "está" el modelo vs si realmente acierta |
| Viola-Jones vs YOLO | Características fijas + AdaBoost vs características aprendidas por una CNN |
| Transformación afín vs homografía | Conserva paralelas (2 × 3) vs conserva rectas pero no paralelas (3 × 3) |

## C. Glosario rápido

| Término | Definición breve |
|---|---|
| AdaBoost | Algoritmo que combina muchos clasificadores débiles en uno fuerte (Viola-Jones) |
| Anchor-free | Detector que predice cajas directamente, sin cajas de referencia predefinidas |
| Backbone / neck / head | Partes de un detector: extraer características / combinar escalas / predecir cajas y clases |
| BGR | Orden azul-verde-rojo en que OpenCV guarda los colores |
| Bilateral | Suavizado que promedia solo píxeles cercanos **y** parecidos; conserva bordes |
| Canny | Detector de bordes en 4 pasos: suavizar, gradiente, adelgazar, histéresis |
| Centroide | Centro de masa de una mancha |
| CLAHE | Ecualización adaptativa por zonas con límite de contraste |
| CNN | Red neuronal convolucional: aprende sus propios kernels |
| COCO | Dataset de 80 clases de objetos cotidianos |
| Confianza | Puntuación del modelo para una predicción (no es probabilidad de acierto) |
| Contorno | Borde exterior de una mancha en una máscara |
| Dilatación | Operación morfológica que engorda las manchas blancas |
| Elemento estructurante | Forma que se usa en morfología (cuadrado, elipse, cruz) |
| Erosión | Operación morfológica que adelgaza las manchas blancas y borra las pequeñas |
| Filtro de Kalman | Predice la siguiente posición de un objeto y la corrige con la medición |
| fps | Cuadros (*frames*) por segundo |
| Frame | Cada imagen de un video |
| Gradiente | Cambio de intensidad (pendiente) en la imagen; base de los bordes |
| Haar (características) | Diferencias entre sumas de rectángulos claros y oscuros |
| Histograma | Cuántos píxeles hay de cada intensidad |
| Homografía | Transformación entre dos vistas de un mismo plano |
| HSV | Espacio de color: tono, saturación, brillo |
| ID switch | Cambio erróneo de identidad de un objeto seguido |
| Imagen integral | Tabla que permite sumar cualquier rectángulo con 4 lecturas |
| Inferencia | Usar un modelo entrenado para predecir |
| IoU | Intersección sobre unión de dos cajas |
| Kernel | Pequeña matriz de pesos que se desliza sobre la imagen |
| Máscara | Imagen binaria (0/255) que indica qué píxeles interesan |
| Mediana | Filtro que toma el valor central ordenado; elimina sal y pimienta |
| MOG2 | Sustractor de fondo que modela cada píxel con varias gaussianas |
| Morfología | Operaciones sobre la forma de las manchas de una máscara |
| NMS | Supresión de no máximos: elimina cajas duplicadas |
| OBB | Caja delimitadora orientada (rotada) |
| ORB | Detector y descriptor de puntos clave, resistente a giros |
| Otsu | Método que elige automáticamente el umbral |
| Padding / stride | Relleno en los bordes / paso con que avanza el kernel |
| Píxel | Unidad mínima de una imagen |
| PSNR | Medida en dB de cuánto se parece una imagen a la original |
| RANSAC | Método robusto que descarta emparejamientos erróneos |
| ROI | Región de interés |
| Segmentación | Asignar a cada píxel una clase o un objeto |
| Sobel | Kernel de derivada que detecta bordes en una dirección |
| Softmax | Convierte puntuaciones en valores entre 0 y 1 que suman 1 |
| Sustracción de fondo | Detectar lo que se mueve comparando con un modelo del fondo |
| Tracking | Seguimiento: mantener la identidad de los objetos entre cuadros |
| Umbral | Valor que separa los píxeles en dos grupos (blanco/negro) |
| Viola-Jones | Detector clásico de rostros (2001) basado en características Haar y cascada |
| YOLO | *You Only Look Once*: familia de detectores de una sola pasada |

## D. Material de apoyo

- [Tutoriales de OpenCV en Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) · [Plataforma interactiva de OpenCV](https://developer.opencv.fr/ui/#/onboard/demo)
- [Animaciones de convolución (conv_arithmetic)](https://github.com/vdumoulin/conv_arithmetic) · [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- [Tareas de Ultralytics](https://docs.ultralytics.com/tasks/) · [YOLO26](https://docs.ultralytics.com/models/yolo26/) · [Seguimiento en Ultralytics](https://docs.ultralytics.com/modes/track/)
- Szeliski, *Computer Vision: Algorithms and Applications* (2.ª ed.), caps. 3 y 9 — gratis en szeliski.org/Book
- Gonzalez y Woods, *Digital Image Processing* (4.ª ed.): capítulos de filtrado espacial y morfología.
- Tutoriales específicos de OpenCV: [Morfología](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) · [Canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html) · [Umbrales](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) · [Sustracción de fondo](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html) · [Cascadas de Haar](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)

**Artículos originales (por si un alumno quiere profundizar):**
- Canny, J. (1986). *A Computational Approach to Edge Detection*. IEEE TPAMI.
- Tomasi, C. y Manduchi, R. (1998). *Bilateral Filtering for Gray and Color Images*. ICCV.
- Viola, P. y Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features*. CVPR.
- Zivkovic, Z. (2004). *Improved Adaptive Gaussian Mixture Model for Background Subtraction*. ICPR.
- Rublee, E. y otros (2011). *ORB: An efficient alternative to SIFT or SURF*. ICCV.
- Krizhevsky, A., Sutskever, I. y Hinton, G. (2012). *ImageNet Classification with Deep Convolutional Neural Networks* (AlexNet). NeurIPS.
- Redmon, J. y otros (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR.
- Zhang, Y. y otros (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV.
- Buolamwini, J. y Gebru, T. (2018). *Gender Shades*. Conference on Fairness, Accountability and Transparency.
