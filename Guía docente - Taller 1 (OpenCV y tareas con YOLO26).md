# Guía docente — Taller 1: Procesamiento de imágenes con OpenCV y tareas de visión con YOLO26

**Notebook:** `Taller01_ProcesamientoImagenesOpencv_Yolo2026/Taller 1_ Procesamiento de imágenes con OpenCV y tareas de visión con YOLO26.ipynb`
**Duración:** 1 h = 15 min de teoría + 45 min de práctica · **Público:** maestría en IA · Colab gratuito, sin GPU

**Objetivos de aprendizaje:**
1. Saber **qué filtro aplicar y en qué momento** del proceso.
2. Procesar **video**: movimiento, seguimiento y conteo, comparando el resultado sin filtros y con filtros.
3. Detectar **rostros con OpenCV** (primer paso del proyecto final).
4. Conocer las **tareas de visión con YOLO26**: detección, segmentación, pose, clasificación y OBB.

**Mensaje central:** "Cada filtro tiene su momento: mejorar contraste → limpiar ruido → analizar → limpiar la máscara. Un sistema puede acertar por casualidad; hay que mirar el proceso, no solo el resultado."

> Todos los valores esperados de esta guía se verificaron ejecutando el notebook. En Colab pueden variar levemente, sobre todo los tiempos.

---

## 1. Teoría (15 min, antes del notebook)

| Min | Tema | Idea para la diapositiva |
|---|---|---|
| 0–3 | Qué es la visión artificial | Conteo de tráfico, control de calidad, seguridad industrial, desbloqueo facial |
| 3–6 | La imagen como datos | Una foto es una tabla de números (alto × ancho × 3 colores); zoom a los píxeles |
| 6–9 | Filtros y convolución | El kernel que se desliza (usar los GIF de la Sección 2); suavizar, enfocar, bordes |
| 9–12 | Del filtro diseñado al aprendido | Las redes convolucionales aprenden sus propios kernels; YOLO = "mira una sola vez" |
| 12–15 | Tareas de visión | Clasificar, detectar, segmentar, pose, OBB (una imagen por tarea) |

---

## 2. Práctica (45 min): sección por sección

### Resumen de tiempos

| Min | Sección | Interactivo |
|---|---|---|
| 0–2 | 0. Configuración | — |
| 2–6 | 1. La imagen como datos y HSV | Sí |
| 6–13 | 2. Filtros: galería, 7 casos y kernel propio | Sí |
| 13–17 | 3. El orden importa | Sí |
| 17–21 | 4. ROI, máscaras y morfología | Sí |
| 21–28 | 5. Video: conteo sin filtros vs con filtros | Sí |
| 28–32 | 6. Rostros con OpenCV | Sí |
| 32–38 | 7. Tareas con YOLO26 | — |
| 38–43 | 8. Zona de seguridad | — |
| 43–45 | Cierre y reto | — |

**Si vas atrasado:** deja el kernel propio (Sección 2) y el anexo para casa, y en la Sección 7 muestra solo la primera figura. **No recortes** las Secciones 3, 5 y 6: son el centro del taller.

---

### Sección 0. Configuración (min 0–2)

**Qué hacer:** pedir que ejecuten **Entorno de ejecución → Ejecutar todas** apenas entren, mientras das la teoría.

**Qué explicar:** nada técnico. Solo avisar:
- La primera celda instala Ultralytics y tarda alrededor de un minuto.
- Si la sesión **se reinicia sola** al instalar, es normal. Ocurre cuando OpenCV ya estaba cargado de una ejecución anterior. Basta con volver a pulsar **Ejecutar todas**.

**Qué deben ver:** "Instalación lista." y luego "Videos listos".

---

### Sección 1. La imagen como datos y HSV (min 2–6)

**Qué explicar:**
- Una imagen es una tabla de números: `img.shape` = (540, 960, 3).
- **OpenCV guarda los colores en orden BGR**, no RGB. La segunda imagen (azulada) es el error típico al mostrarla sin convertir.
- **HSV** separa el color (tono, H) del brillo (V), así que es más fácil "quedarse" con un color. En OpenCV, H va de 0 a 179.

**Qué valores mover (control "Caza colores"):**

| Valores | Resultado | Qué decir |
|---|---|---|
| Inicial: tono 5–20, saturación ≥ 150, brillo ≥ 150 | Detecta los chalecos naranjas, pero en **5 pedazos** (las franjas reflectivas los parten) | "Detecta el color, pero no sabe que son 2 chalecos" (se arregla en las Secciones 2 y 4) |
| **Tono 20–35, saturación ≥ 100, brillo ≥ 150** | **Solo los 2 cascos amarillos** | Solución del reto |
| Tono 20–40, saturación ≥ 80, brillo ≥ 100 | 5 objetos: aparecen zonas amarillentas del fondo | Rangos amplios generan falsas detecciones |
| Tono 5–20, saturación ≥ 50, brillo ≥ 50 | 11 objetos: el piso y la pared entran | La saturación mínima filtra los colores "lavados" |

**Pregunta al grupo:** "¿Qué rompería este detector?" (otro color de chaleco, de noche, un cono naranja).

---

### Sección 2. Filtros: qué hace cada uno y cuándo usarlo (min 6–13)

**Parte A, GIF y galería (2 min).**
- **Qué explicar:** el kernel se desliza, multiplica y suma.
- **Los tres GIF:**
  - Sin relleno: la salida se achica.
  - Con relleno: conserva el tamaño.
  - Saltando de 2 en 2: la salida queda a la mitad (así reducen tamaño las redes).
- **En la galería, señalar tres comparaciones:**
  - El **gaussiano** suaviza todo; la **mediana** deja los bordes más definidos.
  - **Sobel** solo marca bordes verticales; el **Laplaciano**, todos.
  - El **umbral global** se come las sombras; el **adaptativo** dibuja los contornos.

**Parte B, los 7 casos reales (4 min).** Es la parte más importante. Para cada caso, **pregunta primero qué filtro usarían** y después selecciona el caso en el menú.

| Caso (menú "Caso:") | Lo que muestra el panel MAL | Lo que muestra el panel BIEN | Qué decir |
|---|---|---|---|
| 1. Cámara nocturna | Bordes sin filtrar: puro ruido | Gaussiano y luego Canny: contornos limpios | Cada grano parece un borde; suavizar **antes** |
| 2. Interferencia | Gaussiano: los puntos quedan como manchas | Mediana: los puntos desaparecen | La mediana descarta los valores extremos |
| 3. Placa con ruido | Gaussiano fuerte: bordes borrosos | Bilateral: bordes nítidos | El bilateral promedia solo píxeles parecidos; es más lento |
| 4. Contraluz | Ecualización global: quema zonas (REGULAR) | CLAHE: mejora por zonas | CLAHE va **al inicio**; también ayuda a detectar rostros |
| 5. Documento con sombra | Umbral global: el lado oscuro se pierde | Umbral adaptativo: todo legible | Paso previo típico al OCR |
| 6. Contar chalecos | Máscara directa: **17 "objetos"** | Apertura + cierre: **2 objetos** | La morfología va **después** del umbral y **antes** de contar |
| 7. Foto borrosa | Enfoque con ruido: el ruido se amplifica | Enfoque: más nítida | El enfoque va **al final** y solo para mostrar a una persona |

**Parte C, kernel propio (1 min o para casa).** Escribir en el cuadro:

| Kernel | Efecto |
|---|---|
| `0,-1,0,-1,5,-1,0,-1,0` (inicial) | Enfoque |
| `1,1,1,1,1,1,1,1,1` + casilla "Promediar" | Desenfoque |
| `1,1,1,1,1,1,1,1,1` **sin** "Promediar" | Imagen casi blanca: los valores se suman sin dividir (por eso existe la normalización) |
| `-1,-1,-1,0,0,0,1,1,1` | Bordes horizontales |
| `0,1,2,-1,0,1,-2,-1,0` | Bordes diagonales (resalta la cinta amarilla y negra) |
| `0,0,0,0,0,0,0,0,1` | La imagen se desplaza 1 píxel |

---

### Sección 3. El orden importa: ruido → filtro → bordes (min 13–17)

**Qué explicar:** el error más común es buscar bordes directamente sobre una imagen con ruido. Regla: **primero limpiar, después analizar**. El PSNR mide cuánto se parece la imagen a la original (más alto = mejor).

**Qué valores mover, en este orden:**

| Paso | Ruido | Filtro | PSNR | Bordes de Canny | Qué decir |
|---|---|---|---|---|---|
| 1 | Sal y pimienta (30) | **ninguno** (inicial) | 17.6 dB | ~30 % de la imagen son "bordes": puro ruido | Así se ve buscar bordes sin limpiar |
| 2 | Sal y pimienta (30) | gaussiano | 27.3 dB | ~16 %: mejora, pero quedan manchas | Filtro equivocado para este ruido |
| 3 | Sal y pimienta (30) | **mediana** | **32.8 dB** | **~3.5 %**: limpio | Filtro correcto |
| 4 | Gaussiano (30) | ninguno | 18.8 dB | ~37 %: puro ruido | — |
| 5 | Gaussiano (30) | gaussiano | 28.3 dB | ~13 % | Sirve |
| 6 | Gaussiano (30) | **bilateral** | **28.7 dB** | **~4 %**: bordes limpios | El mejor si los bordes importan |

(Como referencia, la imagen sin ruido tiene ~5 % de bordes.)

**Otros controles para mostrar:**
- **Umbrales de Canny:** bajo 10 / alto 30 da textura del piso por todas partes; bajo 150 / alto 250 hace desaparecer contornos de las personas.
- **"Mejorar contraste" con ruido:** el CLAHE también realza el ruido. Por eso, con ruido fuerte, **primero** se limpia.

---

### Sección 4. ROI, máscaras y morfología (min 17–21)

**Qué explicar:**
- **ROI rectangular** = recorte. **ROI poligonal** = máscara con la forma de la zona; se usará con YOLO en la Sección 8.
- `pointPolygonTest`: el punto (700, 450) está dentro de la zona y el (200, 300) fuera.

**Qué valores mover (control "Operación" / "Tamaño").** La máscara de los chalecos empieza con **17 manchas** (lo ideal son 2):

| Operación | Tamaño 5 | Tamaño 9 (inicial) | Tamaño 15 | Tamaño 31 | Qué decir |
|---|---|---|---|---|---|
| Cierre | 14 | 10 | **2** | 2 | El cierre **une** las partes de cada chaleco: con 15 ya son 2 |
| Dilatación | 10 | 6 | **2** | 2 | También une, pero **engorda** el objeto |
| Apertura | 8 | 4 | 4 | 1 | Borra puntitos; con tamaño grande borra objetos completos |
| Erosión | 8 | 4 | 4 | 1 | Adelgaza hasta hacer desaparecer |
| Gradiente | — | — | — | — | Deja solo el borde de cada mancha |

**Pregunta:** "¿Cierre o dilatación para unir?" Respuesta: cierre, porque une sin cambiar tanto el tamaño final.

---

### Sección 5. Video: conteo sin filtros vs con filtros (min 21–28)

**Qué explicar (tabla de 7 pasos del notebook):**
- **Sustracción de fondo:** OpenCV aprende cómo se ve la calle vacía y marca lo que se mueve. Las **sombras salen en gris**.
- Los filtros van en orden: **suavizado antes** de comparar con el fondo → **umbral** para quitar sombras → **apertura** para borrar ruido → **cierre + dilatación** para unir cada auto.
- **Seguimiento por centroides:** cada mancha se une con la más cercana del frame anterior y recibe un ID. Se cuenta cuando cruza la línea roja.

**Qué deben ver (video lado a lado y tabla):**

| | Autos contados (reales: 4) | IDs creados |
|---|---|---|
| Sin filtros | 4 | **40** |
| Con filtros | 4 | **10** |

**Mensaje clave:** sin filtros también cuenta 4, pero **por casualidad**. La máscara es ruido y sombras, y el sistema persigue 40 objetos falsos. Enseña a mirar **la máscara y los IDs**, no solo el número final.

**Qué valores mover (casillas, una a la vez).** Cada cambio procesa el video completo y tarda unos segundos.

| Filtro desactivado | Contados | IDs | Qué decir |
|---|---|---|---|
| Ninguno (todos activos) | 4 | 10 | Referencia |
| Sin suavizado | 5 | 14 | El grano de la cámara genera manchas falsas |
| **Sin quitar sombras** | 3 | **67** | **El filtro más importante:** las sombras se vuelven "objetos" |
| Sin apertura | 3 | 21 | El ruido se cuela en la máscara |
| Sin cierre + dilatación | **6** | 24 | Cada auto se parte en varias manchas y **se cuenta dos veces** |

**Pregunta:** "Si sin filtros cuenta 4, ¿el sistema funciona?" Respuesta: no; acertó por casualidad.

---

### Sección 6. Rostros con OpenCV, Viola-Jones (min 28–32)

**Qué explicar:**
- Detector clásico (2001) que busca patrones de rectángulos claros y oscuros: la zona de los ojos es más oscura que las mejillas. Son kernels como los de la Sección 2.
- Busca rostros de **frente**.
- **Anunciar el proyecto final:** antes de reconocer a alguien, hay que encontrar su rostro.

**Qué valores mover (deslizadores):**

| Frame | Ajuste | Rostros | Qué decir |
|---|---|---|---|
| 488 (inicial) | Valores iniciales | **3 de 3** | Funciona con rostros frontales y cercanos |
| 61 | Valores iniciales | 1 de 1 | — |
| **244** | Valores iniciales (tamaño mínimo 30) | **0 de 2** | Los rostros están lejos: son más pequeños que 30 px |
| **244** | **Tamaño mínimo = 20** | **2 de 2** | Era un problema de **parámetro** |
| 244 | Vecinos mínimos = 1 | 3 (uno es falso) | Pocos vecinos = falsas alarmas |
| 488 | Paso entre escalas = 1.3 | 2 de 3 | Paso grande = más rápido pero se saltan tamaños |
| 488 | Vecinos mínimos = 10 | 2 de 3 | Muy exigente = pierde rostros |
| **305** | Cualquier ajuste razonable | **0 de 1** | Rostro de perfil y movido: es una **limitación del método**, no de los parámetros |
| Cualquiera | "Ecualizar contraste antes" | Sin cambio en este video | Ayuda en escenas oscuras o a contraluz (caso 4 de la Sección 2) |

**Frase de cierre de la sección:** "En el frame 244 bastó ajustar un parámetro. En el 305, el método no da para más: en el Taller 2 usaremos un detector moderno que sí encuentra ese rostro."

---

### Sección 7. Tareas de visión con YOLO26 (min 32–38)

**Qué explicar:**
- Los filtros anteriores **los diseñamos nosotros**; una red convolucional **aprende sus propios filtros** con miles de imágenes.
- YOLO analiza la imagen en **una sola pasada**. YOLO26 (enero 2026) está pensado para funcionar sin GPU.
- Recorrer la tabla de tareas: cada una responde a una pregunta distinta.

**Qué deben ver:**

| Tarea | Modelo | Aprendió con | Resultado |
|---|---|---|---|
| Detección | `yolo26n.pt` | COCO (80 clases) | 2 personas con caja |
| Segmentación | `yolo26n-seg.pt` | COCO | 2 personas pintadas píxel a píxel |
| Pose | `yolo26n-pose.pt` | COCO (17 puntos del cuerpo) | Esqueleto de cada persona |
| Clasificación | `yolo26n-cls.pt` | ImageNet (1000 clases) | Limón: **lemon 51 %**, 2.º *spaghetti squash* 21 % |
| OBB | `yolo26n-obb.pt` | DOTA (imágenes aéreas) | Unos **173 objetos rotados** en la foto del puerto |

**Preguntas al grupo:**
- "¿Qué tarea necesitaría tu proyecto?"
- "¿Por qué el clasificador duda con el limón?" La imagen está movida y el fondo no se parece a las fotos con que aprendió; la confianza no garantiza acierto.

---

### Sección 8. Zona de seguridad con YOLO + seguimiento (min 38–43)

**Qué explicar:**
- Se combinan tres cosas ya vistas: la **zona poligonal** (Sección 4), la **detección de personas** (Sección 7) y el **seguimiento** (versión moderna del tracker de la Sección 5).
- Se usa el **punto de los pies** (centro inferior de la caja), no el centro: con perspectiva, el centro de la caja cae en el torso.

**Qué deben ver:** 2 personas ingresan a la zona en 30 s; el sistema reporta 2 IDs y muestra la alerta en rojo.

**Comparar con la Sección 5:** YOLO **sabe que son personas**, no se confunde con sombras ni con cambios de brillo, y mantiene el ID.

**Experimento opcional:** agregar `tracker='bytetrack.yaml'` dentro de `detector.track(...)`. Aparecen **más IDs** para las mismas 2 personas (en mis pruebas, 6), porque cambian de ID al entrar cortadas por el borde.

---

### Cierre (min 43–45)

- Repasar las 4 ideas del notebook.
- Asignar el reto:
  1. Cascos amarillos (solución: tono 20–35).
  2. Mejor filtro por tipo de ruido.
  3. **Proyecto final:** anotar en qué frames OpenCV pierde rostros.
  4. Probar otra clase o video en la Sección 8.
  5. Cuenta de Roboflow.
- Avance: en el Taller 2, YOLO26 por dentro, cómo entrenarlo con datos propios y reconocimiento facial.

---

### Anexo (para casa): valores sugeridos

**A. Transformaciones:** ángulo 90 con escala 0.5 (rotación pura); inclinación 0.3 (*shear*); "Espejo".

**B. Vista cenital:** solo mirar el GIF. Las líneas del piso quedan en ángulo recto y las columnas se deforman, porque no están en el plano del piso. Los dos trabajadores quedan a 0.50 celdas de distancia.

**C. Plantilla vs ORB.** "Coincidencia" de la plantilla (1 = perfecta) frente a los puntos que coinciden con ORB:

| Giro | Tamaño | Plantilla | ORB | Qué decir |
|---|---|---|---|---|
| 0° | 1.0 | 1.00 (perfecta) | 894 puntos | Ambos aciertan |
| 10° | 1.0 | 0.76 | 733 | La plantilla empieza a fallar con apenas 10° |
| 25° | 0.7 (inicial) | 0.49: marca el lugar equivocado | 499: la encuentra | Diferencia clara |
| 90° | 0.4 | 0.42 | 132: aún la encuentra | ORB resiste giro y tamaño |

---

## 3. Preguntas probables (con respuesta)

- **¿Por qué OpenCV usa BGR?** Por herencia histórica de las cámaras y APIs de su época. Para mostrar con Matplotlib hay que convertir a RGB.
- **¿Gaussiano o mediana?** Gaussiano para ruido granulado; mediana para puntos sueltos (sal y pimienta).
- **¿Siempre hay que suavizar antes de Canny?** Casi siempre. Canny trae un suavizado interno pequeño, pero con ruido fuerte no alcanza.
- **¿Por qué la máscara de los chalecos sale en pedazos?** Las franjas reflectivas no son naranjas. Por eso existe la morfología.
- **¿Por qué Viola-Jones no detecta el rostro de perfil?** El modelo que usamos se entrenó con rostros frontales. OpenCV trae otro para perfiles, pero los detectores modernos lo resuelven mejor.
- **¿YOLO funciona sin GPU?** Sí, la versión nano está pensada para CPU.
- **¿Puedo usar YOLO con mis propias clases?** Sí, ese es el Taller 2 (*fine-tuning*).
- **¿YOLO26 es libre?** Sí, con licencia AGPL-3.0. Para uso comercial cerrado se necesita la licencia Enterprise de Ultralytics.

---

## 4. Riesgos técnicos y plan B

| Riesgo | Plan B |
|---|---|
| `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'` | Ocurre con OpenCV 5 si alguien ejecutó celdas antes de instalar. La celda de instalación reinicia la sesión sola una vez. Si persiste: **Entorno de ejecución → Reiniciar sesión** y ejecutar todo de nuevo. |
| La sesión se reinicia al instalar | Es esperado (ver arriba): volver a pulsar **Ejecutar todas**. |
| Los controles no responden | Ejecutar de nuevo la celda. Se actualizan al **soltar** el deslizador. |
| Las casillas del conteo tardan | Es normal: cada cambio procesa el video completo (unos segundos). |
| Los GIF de convolución no cargan (dependen de GitHub) | Tenerlos en tus diapositivas de teoría. |
| GitHub lento con los videos | Tener la carpeta `material/` en tu Drive y copiarla; el código no descarga lo que ya existe. |
| La zona de seguridad tarda en CPU | Bajar `while n < 1800` a `900` (15 s de video). |
| Colab pisa cambios del notebook | Si editas el archivo fuera de Colab, cierra la pestaña de Colab sin guardar y vuelve a abrirlo desde Drive. |

---

## 5. Checklist antes de clase
- [ ] Ejecutar el notebook completo en una cuenta de Colab gratuita y probar cada control con los valores de esta guía.
- [ ] Compartir el notebook con permiso de **lector**; cada estudiante usa **Archivo → Guardar una copia en Drive**.
- [ ] Preparar las 5 diapositivas de teoría (los GIF de la Sección 2 y las imágenes de la Sección 7 sirven de material).
- [ ] Repasar las tablas de las Secciones 2, 3, 5 y 6: son las que más se preguntan.
- [ ] Pedir la cuenta de Roboflow para el Taller 2.

## 6. Material de apoyo
- [Tutoriales de OpenCV en Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) · [Plataforma interactiva de OpenCV](https://developer.opencv.fr/ui/#/onboard/demo)
- [Animaciones de convolución (conv_arithmetic)](https://github.com/vdumoulin/conv_arithmetic) · [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- [Tareas de Ultralytics](https://docs.ultralytics.com/tasks/) · [YOLO26](https://docs.ultralytics.com/models/yolo26/)
- Szeliski, *Computer Vision: Algorithms and Applications* (2.ª ed.), caps. 3 y 9 — gratis en szeliski.org/Book
