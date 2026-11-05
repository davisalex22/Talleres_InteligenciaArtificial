# Guía docente — Taller 2: YOLO26, entrenamiento, rostros y expresiones faciales

**Notebook:** `Taller 2_ YOLO26, entrenamiento, rostros y expresiones faciales.ipynb`
**Duración:** 1 h · **Modalidad:** clase virtual, explicando sobre el notebook con la pantalla compartida · **Público:** maestría en IA · Colab gratuito **con GPU T4** (para entrenar)

**Objetivos de aprendizaje** (los mismos de la portada del notebook, "Al terminar podrás"):
1. Explicar **cómo decide YOLO** qué objetos mostrar (confianza y cajas repetidas).
2. Combinar **OpenCV y YOLO**: limpiar la imagen antes del modelo y procesar los resultados después.
3. **Entrenar YOLO26 con imágenes propias** (*fine-tuning*), configurar sus **hiperparámetros** y evaluar si funciona bien.
4. Detectar **rostros con sus puntos clave** y reconocer la **expresión facial** de cada persona.

**Mensaje central:** "Un sistema de visión real es una cadena: OpenCV prepara → YOLO detecta → OpenCV alinea → otro modelo clasifica. Ultralytics se controla con parámetros, y cada uno tiene un efecto que se puede medir."

> Los valores de esta guía se verificaron ejecutando el notebook. Los del entrenamiento y la evaluación (Secciones 5 a 8) dependen del dataset de Roboflow y de la GPU: **anótalos en tu ensayo** antes de la clase.

---

## Cómo leer este guion

La guía sigue **el mismo orden y los mismos títulos que el notebook**. Cada sección de la guía es una sección del notebook, y los subtítulos de la guía coinciden con los del notebook. La teoría está dentro de la sección donde el notebook la presenta, así que nunca tienes que saltar a otra parte de la guía.

| Marca | Qué significa |
|---|---|
| **Muestra en pantalla:** | Qué parte del notebook señalar con la pantalla compartida |
| **Haz:** | Acción en el notebook (ejecutar, mover un control, cambiar un valor) |
| **Di:** | Texto para decir en voz alta (adáptalo a tu estilo) |
| **Valores esperados** | Lo que deben ver al ejecutar o al mover cada control |
| **Pregunta al grupo:** | Pregunta para que respondan en el chat o con el micrófono, con la respuesta esperada |
| **Si preguntan:** | Preguntas probables de los alumnos, con respuesta |
| **Transición:** | Frase para pasar a la sección siguiente |

Al final hay apéndices de consulta rápida: plan B técnico, conceptos que se confunden, glosario y material de apoyo.

---

## Antes de clase

- [ ] **Enviar a los estudiantes (unos días antes):** crear la cuenta de Roboflow, copiar su *Private API Key*, crear en Colab el secreto `ROBOFLOW_API_KEY` con *Acceso al notebook* activado, y cambiar el entorno a **GPU T4**. Son los 3 pasos de "Antes de empezar" en la portada del notebook.
- [ ] Ensayar el notebook completo en Colab gratuito con T4. **Anotar:** el número de imágenes por conjunto (Sección 5), el tiempo por época, las métricas globales y la clase con peor resultado (Sección 7), y qué predice cada modelo en la Sección 8.
- [ ] Subir el `best.pt` de tu ensayo a Drive, en `MyDrive/Taller2/best.pt` (es la ruta que usa el Plan B del notebook).
- [ ] Compartir el notebook con permiso de **lector**; cada estudiante usa **Archivo → Guardar una copia en Drive**.
- [ ] Probar la cámara de tu navegador con Colab (Sección 11) y los permisos.
- [ ] Probar compartir la pestaña de Colab y subir el zoom del navegador (Ctrl + +) para que se lean las tablas.
- [ ] Repasar las tablas de valores de las Secciones 2, 4, 9 y 10.

---

## Estructura de la clase (igual que el notebook)

| Min | Parte del notebook | Sección del notebook | Interactivo |
|---|---|---|---|
| 0–4 | Portada | Título, "Al terminar podrás" y "Antes de empezar" | — |
| 0–4 | | **0. Configuración** (se ejecuta mientras hablas de la portada) | — |
| 4–8 | **Parte 1 — Cómo funciona YOLO** | **1. Una sola pasada** | — |
| 8–12 | | **2. Confianza y cajas repetidas** | Sí |
| 12–14 | | **3. Los límites del modelo de fábrica** | — |
| 14–19 | **Parte 2 — YOLO + OpenCV** | **4. OpenCV antes y después del modelo** | Sí |
| 19–22 | **Parte 3 — Entrenamiento con tus propias imágenes** | **5. Dataset propio con Roboflow** | — |
| 22–24 | | **6. Fine-tuning de YOLO26n (en segundo plano)**: lanzar | — |
| 24–34 | | **6. Mientras entrena**: parámetros, aumento de datos, predicción, archivo YAML, progreso | Sí |
| 34–41 | | **7. Evaluación: ¿qué tan bien detecta?** | — |
| 41–44 | | **8. Antes vs después del entrenamiento** y exportar | — |
| 44–47 | **Parte 4 — Rostros y expresiones faciales** | **9. Detección de rostros con puntos clave** | — |
| 47–52 | | **10. Expresiones faciales: YOLO + OpenCV + un segundo modelo** | — |
| 52–56 | | **11. Prueba con tu cámara** y "Para discutir" | Sí |
| 56–60 | Cierre | Lo que te llevas · Reto | — |

**Clave de tiempo:** el entrenamiento corre **en segundo plano**, así que la clase no se detiene a esperarlo: mientras entrena se trabaja la configuración de Ultralytics. Si el grupo va lento, ejecuta tú las Secciones 5 y 6 al inicio de la clase. **No recortes** las Secciones 6, 7 y 10.

---

## Portada del notebook (min 0–4)

**Muestra en pantalla:** la portada: el título, la lista **"Al terminar podrás"** y los 3 pasos de **"Antes de empezar"** (GPU T4, cuenta de Roboflow, secreto `ROBOFLOW_API_KEY`).

**Haz:** comprueba que todos tienen la GPU T4 y el secreto creado, y pide que pulsen **Entorno de ejecución → Ejecutar todas**.

**Di (repaso del Taller 1 y objetivo de hoy):**
> "En el Taller 1 vimos filtros de OpenCV, el detector de rostros de Viola-Jones y las tareas que resuelve YOLO26 con modelos ya entrenados. Hoy vamos a **abrir la caja negra**: entender cómo decide YOLO qué mostrar, combinarlo con OpenCV, **entrenarlo con nuestras propias imágenes** y terminar con rostros y expresiones faciales, que es la base del proyecto final."

**Di (por qué la GPU y Roboflow):**
> "Hoy necesitamos dos cosas que no usamos en el Taller 1. La **GPU T4**, porque vamos a entrenar un modelo, y entrenar son millones de cálculos que la GPU hace en paralelo. Y **Roboflow**, que es de donde vamos a descargar las imágenes con las que entrenamos. La clave de Roboflow se guarda como **secreto** de Colab para no escribirla en el código, donde cualquiera que vea el notebook la podría copiar."

**Si preguntan:**
- **¿Y si no tengo GPU?** El notebook entrena en **modo reducido en CPU**: 2 épocas, imágenes más pequeñas y solo una cuarta parte de los datos. Sirve para ver el proceso, no para obtener buenas métricas.
- **¿El secreto de Roboflow no funciona?** Revisar que *Acceso al notebook* esté activado en el panel de Secretos (ícono de llave). Si falla, el código pide la clave a mano.

**Transición:**
> "Mientras se instala, veamos qué hace la configuración."

---

## 0. Configuración (min 0–4, mientras hablas de la portada)

**Di:**
> "La primera celda instala Ultralytics, que es la biblioteca de YOLO, y Roboflow. Igual que en el Taller 1, si la sesión se reinicia sola al instalar, no rompieron nada: vuelvan a pulsar **Ejecutar todas**."
>
> "La segunda celda descarga todo el material: los videos, la foto del bus, el modelo de rostros, el modelo de expresiones y el detector de OpenCV del Taller 1, para comparar."

**Valores esperados:** "Instalación lista.", el nombre de la GPU (**Tesla T4**) y "Material listo".

**Si preguntan:**
- **¿Qué es `CFG`?** La configuración del entrenamiento. Con GPU: 15 épocas, imágenes de 640 y lotes de 16. Sin GPU: el modo reducido.

**Transición:**
> "Empecemos por entender qué hace YOLO cuando mira una imagen."

---

# Parte 1 — Cómo funciona YOLO

## 1. Una sola pasada (min 4–8)

**Muestra en pantalla:** el texto del notebook con las tres partes de YOLO: **columna (backbone)**, **cuello (neck)** y **cabeza (head)**.

**Di (qué significa "una sola pasada"):**
> "*YOLO* significa *You Only Look Once*, 'solo miras una vez'. Los detectores antiguos miraban la imagen muchas veces: primero buscaban zonas donde *podría* haber algo y después revisaban cada zona por separado. YOLO mira la imagen **una sola vez** y en esa pasada entrega **todas** las cajas, con su clase y su confianza. Por eso es lo bastante rápido para video."

**Di (las tres partes):**
> "Por dentro tiene tres partes, como una línea de producción."
>
> "La **columna**, o *backbone*, son cientos de filtros como los kernels del Taller 1, pero **aprendidos**: nadie los diseñó, el modelo los aprendió viendo miles de imágenes. Extraen bordes, texturas y formas."
>
> "El **cuello**, o *neck*, combina lo que se ve a distintos tamaños, para poder detectar tanto un bus grande como una persona lejana."
>
> "La **cabeza**, o *head*, toma la decisión final: dónde hay un objeto, de qué clase es y con cuánta seguridad."

**Haz:** ejecuta la celda del bus.

**Di (qué devuelve el modelo):**
> "Miren la tabla: esto es **todo** lo que devuelve YOLO. Para cada objeto, tres cosas: la **clase** (qué es), la **confianza** (qué tan seguro está, de 0 a 1) y la **caja**, que son cuatro números: x1 e y1 son la esquina superior izquierda, x2 e y2 la esquina inferior derecha, en píxeles."

**Valores esperados:** la tabla muestra **5 objetos**: **4 personas y el bus**.

**Di (las dos imágenes):**
> "Las dos imágenes se ven iguales, pero la de la izquierda la dibujamos **nosotros con OpenCV**, con un rectángulo y un texto por cada fila de la tabla, y la de la derecha la dibujó Ultralytics con `r.plot()`. El mensaje es: **los resultados de YOLO son solo números**. Con esos números podemos hacer lo que queramos con OpenCV: dibujar, recortar, contar, ocultar. Lo vamos a aprovechar en la Sección 4."

**Si preguntan:**
- **¿Qué es la confianza?** Un número de 0 a 1 que indica qué tan seguro está el modelo de esa detección. No es la probabilidad de acertar: un modelo puede estar muy seguro y equivocarse.
- **¿Por qué las coordenadas están en píxeles?** Porque se refieren a la imagen original; con ellas OpenCV puede recortar directamente la zona del objeto.

**Transición:**
> "Si YOLO mira la imagen una sola vez, ¿cómo decide qué cajas mostrar y cuáles no? Eso es la Sección 2."

---

## 2. Confianza y cajas repetidas (min 8–12)

**Muestra en pantalla:** el texto del notebook con las dos decisiones: **confianza mínima** y **cajas repetidas**.

**Di (confianza mínima):**
> "Para cada objeto, la red propone **muchas cajas candidatas**, cada una con su confianza. La primera decisión es la **confianza mínima**: se descartan las cajas con poca seguridad. Si la bajamos, aparecen más objetos, pero también más errores. Si la subimos, menos errores, pero se pierden objetos."

**Di (el problema: cajas repetidas):**
> "La segunda decisión es sobre las **cajas repetidas**. Cuando la red mira a una persona, no propone una sola caja: propone **varias**, un poco corridas, todas sobre la misma persona. Si las dibujáramos todas, cada persona tendría tres o cuatro cajas encima. YOLO11 y YOLO26 resuelven este problema de forma distinta, y esa es la diferencia principal entre los dos."

**Di (cómo lo resuelve YOLO11: con NMS):**
> "**YOLO11** deja que la red proponga cajas repetidas y después las limpia con un **paso extra**, fuera de la red, llamado **NMS**, *Non-Maximum Suppression*, 'supresión de no máximos'. Piénsenlo como un **filtro de limpieza** que trabaja así, paso a paso:"
>
> 1. "Ordena las cajas de la más segura a la menos segura."
> 2. "Se queda con la **más segura**, la 'máxima'."
> 3. "Revisa las demás: si una se **encima demasiado** con la que se quedó, asume que es el mismo objeto y la **borra**. Eso es 'suprimir las que no son máximas'."
> 4. "Repite con la siguiente caja que sobrevivió, hasta terminar."
>
> "¿Y quién decide qué es 'encimarse demasiado'? El control **'Solapamiento permitido (IoU)'**. Por eso, en YOLO11, el resultado depende de una regla que **nosotros** elegimos, y si la elegimos mal aparecen cajas repetidas o se borran objetos reales."

**Di (cómo lo resuelve YOLO26: sin NMS):**
> "**YOLO26** resolvió el problema **desde el entrenamiento**. Lo entrenaron para que por cada objeto la red entregue **una sola caja**, la mejor. Las repetidas ni siquiera aparecen, así que **no hace falta el paso de limpieza**: lo que sale de la red ya es el resultado final."
>
> "Una comparación: YOLO11 es como un grupo de personas que escriben el mismo informe y después alguien revisa y tacha lo repetido. YOLO26 es como si cada uno ya supiera qué parte le toca: no hay nada repetido que tachar."

**Muestra en pantalla:** los títulos de las dos imágenes del notebook, **"YOLO11n (con NMS)"** y **"YOLO26n (sin NMS)"**.

**Di (por qué importa):**
> "¿Por qué es mejor no necesitar NMS? Por tres razones:"
>
> - "**Más simple:** hay un paso menos y un control menos que ajustar."
> - "**Más rápido en dispositivos pequeños**, como un celular o una cámara: NMS es un paso aparte que consume tiempo, sobre todo cuando hay muchas cajas."
> - "**Más predecible:** el resultado no depende de que elijamos bien el solapamiento."

| | YOLO11 | YOLO26 |
|---|---|---|
| ¿La red propone cajas repetidas? | Sí, varias por objeto | No, una por objeto |
| ¿Cómo se quitan las repetidas? | Con NMS, un paso extra después de la red | No hace falta: aprendió a no repetir |
| ¿Le afecta el control "Solapamiento"? | Sí | No |
| ¿Le afecta el control "Confianza"? | Sí | Sí |

**Haz:** mueve los controles. El notebook propone bajar la **confianza a 0.05** y subir el **solapamiento a 0.95**.

**Valores esperados** (número de cajas: YOLO11 / YOLO26):

| Confianza | Solapamiento 0.3 | Solapamiento 0.7 (inicial) | Solapamiento 0.95 | Di |
|---|---|---|---|---|
| 0.25 (inicial) | 5 / 5 | 5 / 5 | **8 / 5** | "Con solapamiento 0.95, YOLO11 empieza a repetir cajas." |
| **0.05** | 10 / 10 | 11 / 10 | **20 / 10** | "**El ejemplo más claro:** YOLO11 duplica todo; YOLO26 no cambia." |
| 0.5 | 5 / 5 | 5 / 5 | 6 / 5 | "Con confianza alta hay menos cajas candidatas." |

**Di (antes de mover el solapamiento: qué es el IoU):**
> "Para entender este control primero necesitamos saber qué es el **IoU**, *Intersection over Union*, 'intersección sobre unión'. Es un número de 0 a 1 que dice **cuánto se encima una caja sobre otra**. Imaginen dos hojas de papel sobre la mesa:"
>
> - "Si están separadas, no se tocan: IoU = **0**."
> - "Si una tapa más o menos la mitad de la otra: IoU cerca de **0.5**."
> - "Si están exactamente una encima de la otra: IoU = **1**."
>
> "Cuando dos cajas marcan **el mismo objeto**, quedan casi una encima de la otra, con un IoU alto, por ejemplo 0.8. Cuando marcan **dos objetos distintos**, como dos personas una al lado de la otra, se tocan poco, con un IoU bajo."

**Di (qué hace el control de solapamiento):**
> "El control 'Solapamiento permitido' es la **regla que sigue NMS** para decidir si dos cajas son repetidas. NMS compara cada caja con la más segura y se pregunta: '¿se encima más de lo permitido?'. Si **sí**, la considera repetida y la **borra**. Si **no**, la considera otro objeto y la **deja**."
>
> - "Con **0.3**, NMS es **muy estricto**: basta que dos cajas se encimen un poco para borrar una. Quedan pocas cajas, pero hay riesgo de borrar a una persona que está pegada a otra."
> - "Con **0.7**, el valor inicial, es un punto medio: borra las cajas que claramente marcan lo mismo."
> - "Con **0.95**, NMS es **muy permisivo**: solo borra una caja si está **prácticamente encima** de otra. Dos cajas del mismo objeto casi nunca coinciden tanto, así que ya no las reconoce como repetidas y las deja pasar."

**Di (al llegar a confianza 0.05 y solapamiento 0.95):**
> "Miren el título de la imagen de YOLO11: **20 cajas**. Hace un momento, con solapamiento 0.3, eran 10. No aparecieron objetos nuevos en la foto: son **los mismos objetos marcados más de una vez**. Fíjense en la imagen: hay personas y el bus con cajas dobles, casi encima una de la otra. NMS no las borró porque le dijimos que solo borrara las que fueran casi idénticas."
>
> "Ahora miren YOLO26: **10 cajas**, con 0.3, con 0.7 y con 0.95. El control no le hace nada, porque YOLO26 **no usa NMS**: aprendió durante el entrenamiento a dar **una sola caja por objeto**. No hay cajas repetidas que borrar, así que la regla de NMS no tiene a quién aplicarse."
>
> "Un detalle: la **confianza** sí cambia a YOLO26, pasa de 5 a 10 cajas al bajarla a 0.05. Eso es porque la confianza es otra decisión: aparecen objetos de los que está menos seguro. La confianza afecta a los dos modelos; el solapamiento, solo a YOLO11."

**Pregunta al grupo:** "¿Por qué a YOLO26 no le afecta el solapamiento?" (Respuesta: porque no usa NMS; ya da una sola caja por objeto, entonces no hay repetidas que borrar.)

**Si preguntan:**
- **¿Qué es el IoU?** *Intersection over Union*: un número de 0 a 1 que dice cuánto se encima una caja sobre otra. 0 = no se tocan; 1 = idénticas. Como dos hojas de papel sobre la mesa.
- **¿Por qué "no máximos"?** Porque NMS deja la caja **máxima**, la de mayor confianza, y suprime, es decir borra, las que **no** son la máxima y se le encima.
- **¿Qué valor de solapamiento conviene?** El inicial, 0.7, sirve para la mayoría de casos. Se baja si aparecen cajas repetidas, y se sube si hay objetos muy juntos, como personas en una multitud, y NMS está borrando alguno.
- **¿Entonces para qué sirve el control de solapamiento en YOLO26?** Para nada: solo afecta a modelos con NMS. Por eso en la tabla de parámetros de la Sección 6 dice "solo con NMS".
- **¿YOLO26 nunca repite cajas?** Casi nunca, porque lo entrenaron para eso. Es una de sus ventajas: un paso menos, más rápido en dispositivos pequeños.

**Transición:**
> "Ya sabemos cómo decide. Ahora veamos qué pasa cuando le mostramos algo que no aprendió."

---

## 3. Los límites del modelo de fábrica (min 12–14)

**Muestra en pantalla:** la frase del notebook: *"`yolo26n.pt` aprendió 80 clases del dataset COCO. ¿Qué pasa fuera de lo que aprendió?"*

**Di:**
> "Un modelo 'de fábrica' es el que descargamos ya entrenado. `yolo26n.pt` aprendió **80 clases** del dataset COCO: personas, autos, animales, algunas frutas. Veamos qué hace con cosas que no conoce."

**Haz:** ejecuta la celda.

**Valores esperados:**

| Escena | Predicción de `yolo26n.pt` | Di |
|---|---|---|
| Autos vistos desde arriba | *cell phone*, *cell phone* | "En COCO casi no hay autos vistos desde arriba: desde ese ángulo, un auto parece un celular." |
| Limón | *sports ball* | "No conoce el limón: elige lo más parecido de sus 80 clases, una pelota." |
| Piña | *vase* | "Lo mismo: la piña no está, así que dice 'florero'." |
| Personas | *person* ×3 | "Detecta a cada persona, pero COCO **no tiene la clase rostro**." |

**Di (mensaje):**
> "El modelo **solo conoce lo que aprendió**, y nunca dice 'no sé': siempre elige la clase más parecida. Hay dos soluciones, y las vamos a ver hoy: **entrenarlo con nuestras propias clases**, en la Parte 3, o usar un **modelo especializado**, como uno de rostros, en la Parte 4."

**Transición:**
> "Antes de entrenar, veamos cómo OpenCV y YOLO trabajan juntos."

---

# Parte 2 — YOLO + OpenCV

## 4. OpenCV antes y después del modelo (min 14–19)

**Muestra en pantalla:** el texto del notebook: *"YOLO no reemplaza a OpenCV: se complementan."*

**Di:**
> "Una pregunta típica es: si ya tenemos YOLO, ¿para qué OpenCV? La respuesta es que se complementan. **Antes del modelo**, OpenCV **prepara** la imagen: limpia el ruido, recorta una zona, ajusta el tamaño. **Después del modelo**, OpenCV **usa** las cajas: recorta, cuenta, dibuja, oculta, mide."

### Antes del modelo

**Di:**
> "Simulamos una cámara con interferencia: puntos blancos y negros sueltos, el ruido de **sal y pimienta** del Taller 1. Con el valor inicial, 30 % de ruido, YOLO **pierde el bus**. ¿Qué filtro del Taller 1 lo recupera?"

**Pregunta al grupo:** "¿Qué filtro usarían para puntos blancos y negros sueltos?" (Respuesta: la **mediana**.)

**Haz:** mueve el control de ruido y el menú de filtro.

**Valores esperados** (objetos detectados; entre paréntesis, si el bus sigue apareciendo):

| Ruido | Ninguno | Gaussiano | Mediana |
|---|---|---|---|
| 0 % | 5 (con bus) | 5 (con bus) | 5 (con bus) |
| 10 % | 3 (**sin bus**) | 4 (con bus) | 4 (con bus) |
| 20 % | 3 (sin bus) | 3 (sin bus) | 4 (**con bus**) |
| **30 % (inicial)** | 3 (**sin bus**) | 4 (sin bus) | 4 (**con bus**) |
| 40 % | 4 (sin bus) | 4 (sin bus) | 5 (con bus) |

**Di:**
> "El **mismo modelo**, con la **misma imagen**, gana o pierde el bus según el filtro que apliquemos **antes**. Con la mediana lo recupera. Es la regla del Taller 1, 'primero limpiar, después analizar', ahora aplicada a una red neuronal."

### Después del modelo

**Muestra en pantalla:** el texto del notebook: *"con las cajas de YOLO, OpenCV recorta cada persona y la pixela para proteger su privacidad."*

**Haz:** ejecuta la celda.

**Valores esperados:** la foto con las personas **pixeladas** y **4 recortes**, uno por persona.

**Di:**
> "Con los cuatro números de cada caja, OpenCV recorta a cada persona. Esos recortes sirven para dos cosas. Una, **privacidad**: pixelar rostros o placas, como hacen los mapas de calles. Dos, **pasar cada recorte a otro modelo**: eso es exactamente lo que vamos a hacer en la Sección 10 con los rostros y las expresiones."

**Si preguntan:**
- **¿Cómo se pixela?** Se achica el recorte a muy pocos píxeles (8 × 16) y se vuelve a agrandar sin suavizar: quedan cuadrados grandes.
- **¿Siempre ayuda filtrar antes de YOLO?** No siempre: depende del problema. Lo correcto es **medir**, como hicimos con la tabla.

**Transición:**
> "Vimos que el modelo de fábrica no conoce limones ni piñas. Vamos a enseñárselos."

---

# Parte 3 — Entrenamiento con tus propias imágenes

## 5. Dataset propio con Roboflow (min 19–22)

**Muestra en pantalla:** el banner de Roboflow y el texto: *"Roboflow permite anotar imágenes, aumentar datos y exportarlos listos para YOLO."*

**Di (qué es un dataset y qué es Roboflow):**
> "Para enseñarle una clase nueva a YOLO necesitamos un **dataset**: muchas imágenes donde una persona ya marcó, con una caja, dónde está cada objeto y qué es. A eso se le llama **anotar** o **etiquetar**."
>
> "**Roboflow** es una plataforma para hacer eso: subir imágenes, dibujar las cajas, aumentar datos y exportarlos en el formato de YOLO. Además, en **Roboflow Universe** hay miles de datasets públicos. Hoy usamos uno de **frutas**."

**Haz:** ejecuta la celda que descarga el dataset.

**Di (el archivo `data.yaml`):**
> "Lo primero que imprime es el archivo **`data.yaml`**. Es como el índice del dataset: dice dónde están las imágenes y cuáles son los nombres de las clases. El código descarga el formato 'yolov11', que sirve igual para YOLO26."

**Di (entrenamiento, validación y prueba):**
> "El dataset viene dividido en tres conjuntos, y cada uno cumple un rol distinto. Piensen en un estudiante:"
>
> "**Entrenamiento** (*train*): son los ejercicios con los que estudia. El modelo aprende de estas imágenes."
>
> "**Validación** (*valid*): son las pruebas de práctica. Durante el entrenamiento se usan para ver si está mejorando y para elegir el mejor modelo."
>
> "**Prueba** (*test*): es el **examen final**. El modelo **nunca** la ve mientras aprende; se usa una sola vez, al final, para medir de forma honesta."

**Haz:** ejecuta la celda siguiente.

**Valores esperados:** el número de imágenes por conjunto (**anota los valores de tu ensayo**) y **4 imágenes con sus cajas** dibujadas.

**Di (el formato de las etiquetas):**
> "Cada imagen tiene un archivo de texto con **una línea por objeto**: la clase, el centro de la caja, su ancho y su alto. Los valores van de 0 a 1 porque están **en proporción** al tamaño de la imagen: 0.5 significa 'la mitad'. Así la etiqueta sirve aunque la imagen se agrande o se achique. Las cajas verdes que ven las dibujamos con OpenCV a partir de esas líneas."

**Si preguntan:**
- **¿Cuántas imágenes necesito para entrenar?** No hay un número fijo. Como referencia, unos cientos de ejemplos por clase, con variedad de luz, ángulos y fondos. Importa más la variedad que la cantidad.
- **¿Puedo anotar mis propias imágenes?** Sí, en Roboflow se suben y se dibujan las cajas a mano; después se exporta en formato YOLO.

**Transición:**
> "Con el dataset listo, lanzamos el entrenamiento."

---

## 6. Fine-tuning de YOLO26n (en segundo plano) (min 22–34)

**Muestra en pantalla:** el texto del notebook sobre **transferencia de aprendizaje** y el entrenamiento **en segundo plano**.

**Di (qué es el fine-tuning):**
> "No vamos a entrenar desde cero. Partimos de `yolo26n.pt`, que ya sabe ver bordes, texturas y formas, porque aprendió con miles de imágenes de COCO. Solo le enseñamos **nuestras clases**. Eso se llama **transferencia de aprendizaje**, o *fine-tuning*: 'ajuste fino'."
>
> "Es como alguien que ya sabe dibujar y aprende a dibujar frutas: no empieza por aprender a sostener el lápiz. Por eso bastan **pocas imágenes y pocas épocas**."

**Di (qué es una época):**
> "Una **época** es una vuelta completa al dataset: el modelo ve todas las imágenes de entrenamiento una vez. Hoy entrenamos **15 épocas**."

**Haz:** ejecuta la celda de lanzamiento.

**Di:**
> "La celda imprime el comando `yolo detect train ...`: es **el mismo** que escribiríamos en una terminal. Lo lanzamos como un **proceso en segundo plano**, así que el notebook queda libre y podemos seguir trabajando mientras entrena. En un proyecto real es lo normal: el entrenamiento corre en un servidor y se revisa su progreso de vez en cuando."

**Valores esperados:** "Entrenando en segundo plano... sigue con las celdas de abajo." Con la T4, cada época tarda unos **0,3–0,4 min**: **5–6 min** en total.

### Mientras entrena: cómo sacarle provecho a Ultralytics

**Muestra en pantalla:** el texto del notebook: *"Todo en Ultralytics se controla con parámetros (hay más de 100). Se escriben igual en Python y en la terminal."*

**Di:**
> "Un **hiperparámetro** es una perilla que **nosotros** ajustamos antes de entrenar, a diferencia de los pesos, que el modelo aprende solo. En Ultralytics se escriben igual en Python, dentro de `model.train(...)`, y en la terminal, después de `yolo train`."

**Haz:** ejecuta la celda de la tabla de parámetros.

**Di (los que más importan, señalando la tabla):**
> "No hay que aprenderse los 100. Destaco cinco:"
>
> "**`epochs` y `patience`:** cuántas vueltas dar al dataset, y cuántas épocas seguidas sin mejorar esperar antes de detenerse solo. Nuestro comando usa `patience=5`."
>
> "**`imgsz`:** el tamaño al que se lleva cada imagen. Más grande ayuda con objetos pequeños, pero es más lento."
>
> "**`lr0`**, la **tasa de aprendizaje**: qué tan grandes son los ajustes que el modelo hace en cada paso. Muy grande, salta y no se estabiliza; muy pequeña, aprende muy lento. Casi nunca hay que tocarla: `optimizer=auto` elige bien."
>
> "**`freeze`:** congelar las primeras capas, las que ya saben ver bordes y formas, para que no cambien. Útil cuando hay pocas imágenes."
>
> "Y los de **aumento de datos**, que vemos ahora mismo."

#### Aumento de datos: cómo "ve" el modelo tus imágenes al entrenar

**Di (el problema que resuelve):**
> "Tenemos pocas imágenes. Si el modelo ve **siempre las mismas fotos**, en cada época, termina **memorizándolas**: aprende 'la manzana que está en la esquina con luz blanca' en vez de aprender 'qué es una manzana'. Luego le mostramos una foto con otra luz o desde otro ángulo y falla."
>
> "Piensen en cómo reconocemos a un amigo: lo hemos visto de día y de noche, de frente y de perfil, cerca y lejos. Por eso lo reconocemos en cualquier foto. El **aumento de datos**, en inglés *data augmentation*, hace eso con el modelo: le muestra **la misma foto de muchas formas distintas**."

**Di (cómo funciona):**
> "En **cada época**, antes de pasarle una imagen al modelo, Ultralytics le aplica **cambios al azar**: un poco más oscura, girada unos grados, con zoom, en espejo. Cada época el cambio es **distinto**, así que en 50 épocas el modelo ve 50 versiones diferentes de la misma foto, aunque en la carpeta solo haya una."
>
> "Tres cosas importantes:"
>
> - "**Las fotos originales no se modifican:** los cambios se hacen en memoria, solo mientras entrena."
> - "**Las cajas se transforman junto con la imagen:** si la foto se gira o se pone en espejo, la caja de la fruta se mueve con ella. Si no, la etiqueta quedaría en el lugar equivocado y el modelo aprendería mal."
> - "**Solo se aplica en entrenamiento:** en la validación y al usar el modelo, las imágenes se le pasan tal cual."
>
> "Y no es magia: son operaciones de **OpenCV** como las del Taller 1, cambiar a HSV, girar, recortar, voltear."

**Di (las técnicas, una por una, mientras señalas cada control):**

| Control | Técnica | Qué le hace a la imagen | Qué situación real imita | Cuidado |
|---|---|---|---|---|
| `hsv_h` | Tono | Cambia un poco el color: el rojo se va hacia naranja o morado | Cámaras que captan los colores distinto | Si el color define la clase (manzana roja vs verde), dejarlo bajo |
| `hsv_s` | Saturación | Colores más vivos o más apagados, casi grises | Fotos desteñidas o muy coloridas | — |
| `hsv_v` | Brillo | Imagen más clara o más oscura | Fotos de día, de noche, en sombra | — |
| `degrees` | Rotación | Gira la imagen unos grados | Cámara o celular inclinado | Ángulos grandes deforman las cajas y dejan bordes grises |
| `scale` | Zoom | Acerca o aleja | Objetos cerca o lejos de la cámara | — |
| `translate` | Desplazamiento | Mueve la imagen hacia un lado | El objeto no siempre está en el centro | — |
| `fliplr` | Espejo horizontal | Voltea izquierda-derecha | Objetos vistos desde el otro lado | No usar si la dirección importa: letras, flechas, señales de giro |
| `mosaic` | Mosaico | Une 4 imágenes en una | Varios objetos, pequeños y en fondos distintos | — |
| `close_mosaic` | Cerrar mosaico | Apaga el mosaico en las últimas épocas | — | Sirve para que al final el modelo vea fotos normales, como las que verá de verdad |

> "La regla de oro para elegir aumentos: **que imiten lo que la cámara puede ver de verdad**. Si la cámara nunca va a estar de cabeza, no tiene sentido girar 180 grados; eso solo confunde al modelo."

**Si preguntan:**
- **¿El aumento de datos crea más fotos en la carpeta?** No. En Ultralytics los cambios se hacen en memoria, en cada época. El dataset sigue igual.
- **¿Entonces más aumento siempre es mejor?** No. Poco aumento → el modelo memoriza. Demasiado o poco realista → las imágenes ya no se parecen a la realidad y el modelo aprende peor.
- **¿Por qué no se aumenta la validación?** Porque la validación es el "examen": tiene que medir al modelo con fotos reales, sin trucos.
- **¿Cada imagen recibe todos los cambios a la vez?** Cada cambio se aplica al azar, dentro del rango que indica el parámetro. Algunas veces una foto sale casi igual y otras muy cambiada.
- **¿Qué significa el número de cada parámetro?** El **máximo** del cambio: por ejemplo, `degrees = 10` gira al azar entre −10 y +10 grados; `fliplr = 0.5` pone en espejo la mitad de las veces; `mosaic = 1.0` usa mosaico siempre.

**Haz:** muestra el control con los valores iniciales.

**Valores esperados:** valores iniciales: saturación +0.5, brillo 0.3 más oscuro, rotación 10°, zoom 0.2, desplazamiento 0.1 y espejo activado. A la izquierda la imagen original; a la derecha, como la ve el modelo.

**Haz:** mueve la **rotación a 45°**.

**Di:**
> "Con 45 grados las cajas se deforman y aparecen bordes grises: eso confunde al modelo. Por eso Ultralytics trae la rotación en 0 por defecto. Los aumentos tienen que parecerse a lo que la cámara puede ver de verdad."

**Haz:** activa **mosaic**.

**Di:**
> "**Mosaic** une 4 imágenes en una. Así el modelo ve objetos más pequeños y en contextos variados. Está activo por defecto."

**Haz:** ejecuta la celda de `train_batch0.jpg` (si aún no aparece, vuelve a ejecutarla en unos segundos).

**Di:**
> "Esto **no es una simulación**: es lo que **realmente** vio el modelo en su primer lote de entrenamiento. Fíjense en los mosaicos, los cambios de color y las cajas que se movieron con las frutas."

#### Parámetros de predicción

**Muestra en pantalla:** el texto del notebook: *"Al usar el modelo también se configura."* y el frame 244 del pasillo (2 personas lejanas).

**Haz:** cambia `imgsz`.

**Valores esperados:**

| imgsz | Personas detectadas (hay 2) | Tiempo (CPU de prueba) |
|---|---|---|
| 160 | **0** | ~6 ms |
| 320 | 2 | ~9 ms |
| 640 (inicial) | 2 | ~23 ms |
| 1280 | 2 | ~75 ms |

**Di:**
> "Con `imgsz` 160 la imagen se achica tanto que las personas lejanas desaparecen. Con 1280 las encuentra, pero tarda mucho más. Es un **compromiso entre detectar objetos pequeños y velocidad**. Mientras entrena, los tiempos pueden ser mayores, porque comparte la máquina."

**Haz:** baja `max_det` a 1 y luego activa `classes=[0]`.

**Di:**
> "**`max_det`** limita cuántos objetos devuelve: con 1, solo uno. **`classes=[0]`** filtra: solo personas, la clase 0 de COCO. Así ignoramos lo que no nos interesa."

#### Guardar tu configuración en un archivo

**Haz:** ejecuta la celda que crea `output/mi_config.yaml`.

**Di:**
> "En vez de escribir los parámetros cada vez, se guardan en un archivo **YAML**, que es un archivo de texto con 'nombre: valor' en cada línea. Después se usa con `cfg=`. Así queda registrado con qué configuración se entrenó cada modelo. No lo ejecutamos ahora porque ya hay un entrenamiento en curso."

**Muestra en pantalla:** las "Otras opciones útiles" del notebook.

**Di:**
> "Tres opciones más: **`resume=True`** continúa un entrenamiento que se cortó, desde el último punto guardado. **`freeze=10`** congela las primeras 10 capas. Y **`model.tune()`** hace una **búsqueda automática** de hiperparámetros: entrena muchas veces cambiando los valores y se queda con los mejores. Es muy costoso, cientos de entrenamientos, así que es para un proyecto, no para la clase."

#### Progreso del entrenamiento

**Haz:** ejecuta `progreso()`.

**Di (qué es la pérdida):**
> "La **pérdida**, o *loss*, mide **cuánto se equivoca** el modelo: tiene que **bajar**. El gráfico de la izquierda es el error al **ubicar las cajas**. El de la derecha es el **mAP**, la calidad general del detector, que lo veremos en la Sección 7: tiene que **subir**."
>
> "Hay una señal de alarma que deben conocer: si la pérdida de **entrenamiento** baja pero la de **validación** sube, el modelo está **memorizando** las fotos en vez de aprender. Se llama **sobreajuste**, u *overfitting*: es como el estudiante que memoriza las respuestas de los ejercicios y luego falla el examen."

**Haz:** ejecuta la celda de seguimiento época a época.

**Valores esperados:** una fila por cada época terminada, con estas columnas:

| época | min | box_loss | cls_loss | precisión | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|

**Di (mientras aparecen las filas):**
> "**`box_loss`** es el error al ubicar la caja y **`cls_loss`**, el error al decir la clase: los dos **bajan**. Precisión, recall y mAP **suben**. La columna `min` dice cuánto va tardando. Al terminar imprime el tiempo total y el gráfico."

**Muestra en pantalla:** el **Plan B** del notebook (si falla la GPU o se desconecta la sesión, se carga un `best.pt` desde Drive).

**Si preguntan:**
- **¿Qué hiperparámetros conviene tocar primero?** `epochs` y `patience`, luego `imgsz` (si hay objetos pequeños) y el aumento de datos según la escena. `lr0` y `optimizer` rara vez hace falta cambiarlos.
- **¿Por qué el notebook borra la carpeta `runs/frutas` antes de entrenar?** Para que no se mezclen los resultados de un entrenamiento anterior con el nuevo.
- **¿Por qué la celda de predicción usa otro modelo (`modelo_pred`)?** Porque Ultralytics **recuerda** el filtro `classes` entre llamadas; con una instancia aparte no afecta al resto del notebook.
- **¿`model.tune()` siempre mejora?** Suele mejorar unos puntos de mAP, pero cuesta muchas horas de GPU. Primero hay que mejorar los datos.
- **¿Qué es `best.pt` y `last.pt`?** `last.pt` es el modelo de la última época; `best.pt`, el de la época con mejores resultados en validación. Para usar el modelo se toma `best.pt`.
- **¿Qué es un lote (`batch`)?** Cuántas imágenes ve el modelo antes de hacer un ajuste. Usamos 16.

**Transición:**
> "El entrenamiento terminó. Ahora la pregunta importante: ¿qué tan bien detecta?"

---

## 7. Evaluación: ¿qué tan bien detecta? (min 34–41)

**Muestra en pantalla:** el texto del notebook: *"Evaluamos con las imágenes de prueba, que el modelo nunca vio."*

**Di (por qué con la prueba):**
> "Evaluamos con el conjunto de **prueba**, el examen final. El mejor modelo se eligió mirando la validación, así que medir con la validación sería hacer trampa. La prueba es una medición honesta, con imágenes que el modelo nunca vio."

**Di (cuándo una detección es correcta):**
> "Primero hay que decidir cuándo una caja está bien. Se usa el **IoU**, que ya vimos: cuánto coincide la caja del modelo con la caja real, de 0 a 1. Si coinciden lo suficiente, por ejemplo más de la mitad, y la clase es correcta, es un **acierto**."
>
> "Con eso, cada caja cae en uno de tres grupos: **verdadero positivo**, un acierto; **falso positivo**, una falsa alarma, detectó algo que no estaba o con la clase equivocada; y **falso negativo**, un objeto real que se le escapó."

**Muestra en pantalla:** el **ejemplo de las naranjas** y la tabla de métricas del notebook.

**Di (el ejemplo, sin fórmulas):**
> "En las imágenes de prueba hay **10 naranjas**. El modelo dibuja **8 cajas** de naranja: **6 están bien** y **2 son errores**, una manzana y un trozo de fondo. Y se le escaparon **4** naranjas."
>
> "**Precisión:** de lo que detectó, ¿cuánto era correcto? 6 de 8, **75 %**. Mide las **falsas alarmas**."
>
> "**Recall**, o sensibilidad: de lo que había, ¿cuánto encontró? 6 de 10, **60 %**. Mide los **objetos perdidos**."
>
> "**F1:** junta las dos en un solo número de equilibrio: **67 %**."
>
> "**AP**, precisión promedio: la calidad de **una clase**, probando todos los umbrales de confianza posibles."
>
> "**mAP50:** el promedio del AP de **todas las clases**, contando como acierto las cajas con IoU de 0.5 o más. Y **mAP50-95** es lo mismo, pero cada vez más exigente con qué tan bien ajustada está la caja: es la métrica más estricta."

**Di (por qué no la *accuracy*):**
> "Quizá esperaban la **accuracy**, el porcentaje de aciertos sobre el total. Se usa en **clasificación**, una etiqueta por imagen, como el modelo de expresiones de la Sección 10. Pero en detección no sirve: en una imagen hay infinitas zonas de fondo vacío 'correctamente no detectadas', y no se pueden contar. Por eso los detectores se miden con precisión, recall y mAP."

**Di (precisión contra recall):**
> "Precisión y recall dependen de la **confianza mínima**. Si la subo, el modelo se equivoca menos, sube la precisión, pero se le escapan más objetos, baja el recall. ¿Cuál importa más? Depende del problema. En una **alarma de seguridad** importa no perder a nadie: recall. En un **control de calidad**, no rechazar piezas buenas: precisión."

**Haz:** ejecuta la celda de evaluación.

**Valores esperados:** una tabla con precisión, recall, F1, AP50 y AP50-95 **por clase** (ordenada de peor a mejor) y una línea **Global** con precisión, recall, mAP50 y mAP50-95. **Usa los valores de tu ensayo.**

**Pregunta al grupo** (la del notebook): "¿Qué clase tiene el peor resultado? ¿El problema es de precisión (falsas alarmas) o de recall (objetos perdidos)?"

### Matriz de confusión

**Muestra en pantalla:** la imagen explicativa de la matriz de confusión del notebook.

**Di:**
> "La **matriz de confusión** cruza lo que **era** cada objeto con lo que el modelo **dijo** que era. La diagonal son los aciertos; lo que está fuera, las confusiones: por ejemplo, naranjas que dijo que eran limones."
>
> "En detección tiene una fila y una columna extra, **background**, fondo. La **columna background** son objetos reales que el modelo **no encontró**. La **fila background** son detecciones donde **no había nada**: falsas alarmas."

**Haz:** ejecuta la celda de la matriz y las curvas.

**Di (las imágenes):**
> "Después vemos lado a lado la **anotación real** y la **predicción del modelo** en un lote de prueba: es la forma más rápida de ver dónde se equivoca."
>
> "La **curva precisión-recall** muestra el equilibrio entre las dos a medida que cambia la confianza: cuanto más área encierra, mejor la clase. La **curva F1** muestra qué confianza mínima da el mejor equilibrio."

**Pregunta al grupo:** "¿Qué harían para mejorar la clase con peor resultado?" (Respuesta: más imágenes de esa clase, mejores anotaciones o más épocas.)

**Si preguntan:**
- **¿Qué es un buen mAP?** Depende del problema y del dataset; sirve sobre todo para **comparar** modelos entrenados con los mismos datos.
- **¿Por qué el mAP50-95 siempre es más bajo que el mAP50?** Porque exige cajas mucho más ajustadas.

**Transición:**
> "Los números están bien, pero veamos el cambio con nuestros propios ojos."

---

## 8. Antes vs después del entrenamiento (min 41–44)

**Muestra en pantalla:** el texto del notebook: *"Volvemos a la banda transportadora: modelo de fábrica (COCO) contra nuestro modelo entrenado."*

**Haz:** ejecuta la celda.

**Valores esperados:** 3 frames de la banda (900, 1020 y 3180), cada uno con el modelo de fábrica a la izquierda y el entrenado a la derecha. **Usa lo que anotaste en tu ensayo.**

**Di:**
> "A la izquierda, el modelo de fábrica, que confundía el limón con una pelota. A la derecha, el nuestro: ahora reconoce **las clases del dataset**."
>
> "Pero ojo, como dice el notebook: el modelo entrenado solo conoce las clases **de su dataset**. Si una fruta no estaba en las imágenes de entrenamiento, la confunde con la más parecida. **Los datos definen lo que el modelo puede ver.**"

### Llevar el modelo a producción: exportar

**Di (qué es exportar):**
> "Para usar el modelo fuera de este notebook, por ejemplo en una aplicación, un celular o una cámara, se **exporta**. `export` convierte el modelo a otros formatos que funcionan **sin PyTorch**."
>
> "El más usado es **ONNX**, un formato estándar: es como el PDF de los modelos, lo abren muchos programas, incluso OpenCV con su módulo `cv2.dnn`. Hay otros según dónde se va a usar: **TensorRT** para GPU NVIDIA, **OpenVINO** para procesadores Intel y **TFLite** para celulares."

**Haz:** ejecuta la celda de exportación.

**Valores esperados:** el tiempo por imagen en CPU de PyTorch y de ONNX. En la prueba, ONNX fue algo más rápido (unos 7 vs 8 ms); en Colab los tiempos serán otros.

**Si preguntan:**
- **¿Por qué ONNX es más rápido?** Porque está optimizado solo para predecir; no carga todo lo necesario para entrenar.

**Transición:**
> "Pasemos a la última parte, la más cercana al proyecto final: rostros."

---

# Parte 4 — Rostros y expresiones faciales

## 9. Detección de rostros con puntos clave (min 44–47)

**Muestra en pantalla:** el texto del notebook sobre **YOLO26n-face**.

**Di:**
> "COCO no tiene la clase 'rostro', así que usamos **YOLO26n-face**: un YOLO26 que la comunidad entrenó con miles de rostros. Es **fine-tuning**, exactamente lo que hicimos con las frutas, pero con caras."
>
> "Además de la caja, entrega **5 puntos clave**: los dos ojos, la nariz y las dos comisuras de la boca. Un **punto clave** es una posición concreta que el modelo aprendió a ubicar, como las articulaciones del modelo de pose del Taller 1."

**Haz:** ejecuta la celda que compara con Viola-Jones.

**Valores esperados:**

| Frame | Escena | OpenCV (Viola-Jones) | YOLO26n-face |
|---|---|---|---|
| 244 | 2 rostros pequeños | 0 | **2** |
| 305 | Perfil y movido | 0 | **1** |
| 488 | 3 frontales | 3 | 3 |

**Di:**
> "¿Se acuerdan del reto del Taller 1, anotar en qué frames OpenCV perdía rostros? Aquí está la respuesta: en el 244, los rostros pequeños, y en el 305, el de perfil. YOLO26n-face encuentra los dos, y además marca los 5 puntos de cada uno. Esos puntos son la clave de la siguiente sección."

**Transición:**
> "Ya encontramos los rostros. Ahora vamos a saber qué expresión tienen."

---

## 10. Expresiones faciales: YOLO + OpenCV + un segundo modelo (min 47–52)

**Muestra en pantalla:** la tabla de 3 pasos del notebook.

**Di (la cadena completa):**
> "Para reconocer la expresión unimos todo lo aprendido en una cadena de tres pasos:"
>
> "**Paso 1, YOLO26n-face:** encuentra cada rostro y sus 5 puntos clave."
>
> "**Paso 2, OpenCV:** usa los puntos clave para **enderezar** el rostro y llevarlo a 112 por 112 píxeles."
>
> "**Paso 3, el modelo de expresiones:** clasifica el rostro en 7 expresiones: enojo, asco, miedo, felicidad, neutral, tristeza y sorpresa."

**Di (qué es alinear y por qué):**
> "**Alinear** es como poner todas las fotos carné en la misma posición: OpenCV gira, escala y mueve cada rostro para que los ojos, la nariz y la boca queden **siempre en el mismo lugar**. Es la misma idea de las transformaciones del anexo del Taller 1."
>
> "¿Por qué hace falta? Porque el modelo de expresiones aprendió con rostros alineados. Si le damos una cabeza inclinada, se confunde."

**Di (el segundo modelo):**
> "El modelo de expresiones es una red **MobileFaceNet** del **OpenCV Model Zoo**, una colección de modelos libres. Es un modelo de **clasificación**: no busca dónde está nada, solo responde '¿qué expresión es?', con una probabilidad para cada una de las 7. Y se ejecuta con **OpenCV**, con `cv2.dnn`, sin PyTorch: es el mismo tipo de uso que vimos al exportar a ONNX."

**Haz:** ejecuta la celda.

**Valores esperados:** en el **frame 900**, **2 rostros**, ambos **"felicidad" al 100 %**; los recortes alineados de cada rostro; y un gráfico de barras con la probabilidad de cada expresión del primer rostro.

**Di (el gráfico):**
> "Las barras son las probabilidades de las 7 expresiones: siempre suman 100 %. El modelo elige la más alta. Aquí no tiene dudas."

**Haz:** ejecuta la celda del GIF.

**Valores esperados:** las dos personas pasan de **neutral → sorpresa → felicidad → neutral** (frames 830 a 970).

**Di:**
> "Es la misma cadena aplicada cuadro a cuadro a un video. Miren cómo cambia la etiqueta y el color de la caja con la expresión."

**Pregunta al grupo:** "¿Por qué alinear antes de clasificar?" (Respuesta: el modelo aprendió con rostros alineados; sin alinear, una cabeza inclinada lo confunde.)

**Si preguntan:**
- **¿El modelo de expresiones también se entrenó con fine-tuning?** Es una red MobileFaceNet entrenada con el dataset RAF-DB, fotos reales de expresiones. Se podría ajustar con fotos propias, como hicimos con las frutas.
- **¿Qué cambia para el proyecto final?** Solo el **paso 3**: en vez de "¿qué expresión tiene?", "¿quién es?". Se usa un modelo de **reconocimiento facial**, que compara cada rostro alineado con una galería de personas registradas (por ejemplo SFace, también del OpenCV Model Zoo).

**Transición:**
> "Ahora pruébenlo con su propia cara."

---

## 11. Prueba con tu cámara (min 52–56)

**Muestra en pantalla:** las instrucciones del notebook y la nota: *"Solo con tu consentimiento. La imagen se procesa en tu sesión de Colab y no se guarda en ningún otro lugar."*

**Di:**
> "Ejecuten la celda, permitan el acceso a la cámara y pulsen **Capturar**. Prueben distintas caras: sonrían, sorpréndanse, pongan cara seria o triste. Si prefieren no usar la cámara, cambien `USAR_CAMARA = False` y suban una foto. Es **voluntario**."

**Haz:** haz tú también la prueba con la pantalla compartida y pide que escriban en el chat qué expresión les detectó.

**Valores esperados:**
- Cada rostro con su expresión y las 3 más probables.
- **Felicidad, sorpresa y neutral** se reconocen muy bien.
- **Tristeza, enojo, asco y miedo** son más difíciles y a veces se confunden entre sí.
- Fuera de Colab, la celda usa el frame 858 del video: 2 rostros con "sorpresa" al 100 %.

### Para discutir

**Muestra en pantalla:** la celda **"Para discutir"** del notebook.

**Di:**
> "Tres puntos para pensar."
>
> "**Uno:** una **expresión** no es lo mismo que una **emoción**. Alguien puede sonreír estando nervioso. El modelo solo ve la cara, no lo que la persona siente."
>
> "**Dos:** el modelo aprendió con fotos de un tipo de personas, luces y ángulos. ¿Funciona igual con todas las personas? Los **sesgos** de los datos se vuelven sesgos del modelo."
>
> "**Tres, la regulación:** en la Unión Europea, el **AI Act** prohíbe usar reconocimiento de emociones en **lugares de trabajo y centros educativos**, salvo por razones médicas o de seguridad. En Ecuador, la **Ley Orgánica de Protección de Datos Personales** considera los datos biométricos como **datos sensibles**."

**Pregunta al grupo:** "¿Por qué creen que la detección de felicidad funciona mejor que la de miedo o asco?" (Respuesta esperada: son expresiones más marcadas y más frecuentes en las fotos de entrenamiento.)

**Si preguntan:**
- **¿Puedo usar esto para vigilar a mis estudiantes o empleados?** En la UE está prohibido en educación y trabajo, y en Ecuador requiere base legal y consentimiento. Es un ejemplo didáctico.
- **La cámara no abre.** Revisar los permisos del navegador; si no funciona, `USAR_CAMARA = False` y subir una foto.

**Transición:**
> "Cerremos con lo que se llevan."

---

## Cierre (min 56–60)

### Lo que te llevas

**Muestra en pantalla:** la celda **"Cierre"** del notebook.

**Di:**
> "Cuatro ideas."
>
> "**Uno:** YOLO mira la imagen **una sola vez**; la **confianza** decide qué se muestra, y YOLO26 ya no necesita eliminar cajas repetidas."
>
> "**Dos:** **OpenCV y YOLO se complementan**: OpenCV limpia la imagen antes y usa las cajas después, para recortar, ocultar o alinear."
>
> "**Tres:** con **fine-tuning**, un modelo aprende **nuestras clases** con pocas imágenes y pocas épocas. Sus hiperparámetros se configuran en Python, en la terminal o en un archivo YAML, y se evalúa con precisión, recall y mAP."
>
> "**Cuatro:** la cadena **detectar → alinear con puntos clave → clasificar** reconoce expresiones. **Cambiando el último modelo** por uno de reconocimiento facial, la misma cadena reconoce **quién es** la persona. Ese es su proyecto final."

### Reto

**Muestra en pantalla:** la lista del **Reto** del notebook y léela con ellos.

1. En la Sección 2, ¿con qué combinación de confianza y solapamiento YOLO11 repite más cajas? (Respuesta: confianza 0.05 y solapamiento 0.95 → 20 cajas.)
2. En la Sección 4, ¿desde qué porcentaje de ruido YOLO pierde el bus sin filtro? (Respuesta: desde el 10 %.)
3. Entrenar con otro dataset de Roboflow Universe del área de cada uno y comparar las métricas.
4. Entrenar de nuevo con `output/mi_config.yaml` y comparar el mAP con el primer modelo.
5. **(Proyecto final)** Llevar el análisis de expresiones a video con `vid_stride` y la cámara.

---

# APÉNDICES (consulta rápida)

## A. Si algo falla: riesgos técnicos y plan B

| Riesgo | Plan B |
|---|---|
| No hay GPU T4 disponible | El notebook entrena en modo reducido en CPU (2 épocas, imágenes de 320, 25 % de los datos). Para la evaluación, usar el `best.pt` del Plan B. |
| El secreto de Roboflow no funciona | El código pide la clave a mano. Verificar *Acceso al notebook* en el panel de Secretos. |
| La sesión se reinicia al instalar | Esperado (corrección de OpenCV 5): volver a **Ejecutar todas**. |
| El entrenamiento tarda más de lo previsto | Bajar `epochs` a 10 en `CFG`, o usar el Plan B. |
| El entrenamiento en segundo plano falla | La celda de seguimiento muestra el final de `output/entrenamiento.log`. Error típico: falta la clave de Roboflow o la ruta de `data.yaml`. |
| Se desconecta la sesión durante el entrenamiento | Usar la celda del **Plan B**: carga `best.pt` desde `MyDrive/Taller2/` y continuar con la Sección 7. |
| Las celdas interactivas van lentas mientras entrena | Es normal: comparten la máquina con el entrenamiento. Usar valores de `imgsz` pequeños. |
| La cámara no abre | Revisar los permisos del navegador; si no, `USAR_CAMARA = False` y subir una foto. |
| Un estudiante no quiere usar su cámara | Usar una foto que no sea de una persona real del grupo; es voluntario. |
| URLs externas caídas (modelos o videos) | Tener la carpeta `material/` en tu Drive y copiarla; el código no descarga lo que ya existe. |

## B. Conceptos que se confunden

| Par de conceptos | Diferencia en una línea |
|---|---|
| Confianza vs exactitud | Lo seguro que "está" el modelo vs si realmente acierta |
| Confianza mínima vs solapamiento (IoU) | Descarta cajas dudosas vs descarta cajas repetidas (solo con NMS) |
| YOLO11 vs YOLO26 | Elimina cajas repetidas con NMS vs aprendió a no repetirlas |
| Entrenar desde cero vs fine-tuning | Aprender todo vs reutilizar lo que el modelo ya sabe y enseñarle clases nuevas |
| Parámetro (peso) vs hiperparámetro | Lo aprende el modelo vs lo elegimos nosotros antes de entrenar |
| Época vs lote (batch) | Una vuelta completa al dataset vs las imágenes que ve antes de cada ajuste |
| Entrenamiento vs validación vs prueba | Ejercicios para estudiar vs pruebas de práctica vs examen final |
| Pérdida (loss) vs mAP | Cuánto se equivoca (debe bajar) vs qué tan bien detecta (debe subir) |
| Precisión vs recall | Pocas falsas alarmas vs pocos objetos perdidos |
| mAP50 vs mAP50-95 | Acepta cajas aproximadas vs exige cajas cada vez más ajustadas |
| Accuracy vs mAP | Métrica de clasificación vs métrica de detección |
| `best.pt` vs `last.pt` | El mejor modelo en validación vs el de la última época |
| Detectar rostros vs reconocer expresiones vs reconocer personas | ¿Dónde hay una cara? vs ¿qué expresión tiene? vs ¿de quién es? |
| Expresión vs emoción | Lo que muestra la cara vs lo que siente la persona |

## C. Glosario rápido

| Término | Definición breve |
|---|---|
| Alinear (un rostro) | Girar, escalar y mover el rostro para que ojos, nariz y boca queden siempre en el mismo lugar |
| Aumento de datos | Modificar al azar las imágenes en cada época (color, rotación, zoom, espejo, mosaico) para que el modelo no memorice |
| Backbone / neck / head | Columna (extrae bordes y formas) / cuello (combina tamaños) / cabeza (decide cajas, clase y confianza) |
| `batch` (lote) | Cuántas imágenes ve el modelo antes de hacer un ajuste |
| `best.pt` | El modelo de la época con mejores resultados en validación |
| Caja (x1, y1, x2, y2) | Esquina superior izquierda e inferior derecha del objeto, en píxeles |
| COCO | Dataset de 80 clases de objetos cotidianos con el que se entrenó `yolo26n.pt` |
| Confianza | Qué tan seguro está el modelo de una detección (0 a 1) |
| `cv2.dnn` | Módulo de OpenCV para ejecutar redes neuronales sin PyTorch |
| `data.yaml` | Archivo índice del dataset: rutas de las imágenes y nombres de las clases |
| Época | Una vuelta completa al dataset de entrenamiento |
| Exportar | Convertir el modelo a otro formato (ONNX, TensorRT, OpenVINO, TFLite) para usarlo fuera de PyTorch |
| F1 | Equilibrio entre precisión y recall en un solo número |
| Fine-tuning | Ajustar un modelo ya entrenado para que aprenda clases nuevas |
| `freeze` | Congelar las primeras capas para que no cambien al entrenar |
| Hiperparámetro | Valor que elegimos antes de entrenar (épocas, tamaño de imagen, tasa de aprendizaje…) |
| `imgsz` | Tamaño al que se lleva la imagen; más grande detecta objetos pequeños, pero es más lento |
| IoU | Cuánto se solapan dos cajas (0 = nada, 1 = idénticas) |
| `lr0` | Tasa de aprendizaje: tamaño de los ajustes en cada paso |
| mAP50 / mAP50-95 | Calidad general del detector, promediando todas las clases |
| Matriz de confusión | Tabla que cruza la clase real con la predicha; en detección incluye "background" |
| MobileFaceNet | Red pequeña que usamos para clasificar 7 expresiones faciales |
| Mosaic | Aumento de datos que une 4 imágenes en una |
| NMS | Paso que borra cajas repetidas sobre el mismo objeto (YOLO11 lo usa; YOLO26 no) |
| ONNX | Formato estándar de modelos, como el "PDF" de las redes neuronales |
| Overfitting (sobreajuste) | El modelo memoriza las fotos de entrenamiento y falla con fotos nuevas |
| `patience` | Épocas sin mejorar antes de detener el entrenamiento |
| Pérdida (loss) | Cuánto se equivoca el modelo; debe bajar al entrenar |
| Precisión | De lo que detectó, cuánto era correcto |
| Punto clave | Posición concreta que el modelo ubica (ojos, nariz, comisuras de la boca) |
| Recall | De lo que había, cuánto encontró |
| Roboflow | Plataforma para anotar, aumentar y exportar datasets |
| Transferencia de aprendizaje | Reutilizar lo que un modelo ya sabe para una tarea nueva |
| YAML | Archivo de texto con "nombre: valor" para guardar configuraciones |
| YOLO26n-face | YOLO26 entrenado por la comunidad para detectar rostros y 5 puntos clave |

## D. Material de apoyo

- [Ultralytics: entrenamiento](https://docs.ultralytics.com/modes/train/) · [Métricas de YOLO](https://docs.ultralytics.com/guides/yolo-performance-metrics/) · [YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [YOLO-face (akanametov)](https://github.com/akanametov/yolo-face) · [Expresiones faciales, OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition) · [SFace, OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
- [Roboflow Universe](https://universe.roboflow.com/)
