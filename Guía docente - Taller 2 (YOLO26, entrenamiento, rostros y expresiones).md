# Guía docente — Taller 2: YOLO26, entrenamiento, rostros y expresiones faciales

**Notebook:** `Taller02_YOLO26_Entrenamiento_Rostros_Emociones/Taller 2_ YOLO26, entrenamiento, rostros y expresiones faciales.ipynb`
**Duración:** 1 h = 15 min de teoría + 45 min de práctica · **Público:** maestría en IA · Colab gratuito **con GPU T4** (para entrenar)

**Objetivos de aprendizaje:**
1. Entender **cómo decide YOLO** qué mostrar (confianza, cajas repetidas, NMS).
2. Ver cómo **OpenCV y YOLO se complementan** (antes y después del modelo).
3. **Entrenar YOLO26 con datos propios** (*fine-tuning*), **configurar sus hiperparámetros** y evaluarlo.
4. Detectar **rostros con puntos clave** y reconocer **expresiones faciales**, base del proyecto final.

**Mensaje central:** "Un sistema de visión real es una cadena: OpenCV prepara → YOLO detecta → OpenCV alinea → otro modelo clasifica. Ultralytics se controla con parámetros (en Python, terminal o YAML) y cada uno tiene un efecto que se puede medir."

> Los valores de esta guía se verificaron ejecutando el notebook. Los del entrenamiento (Sección 6–8) dependen del dataset real de Roboflow y de la GPU: **anótalos en tu ensayo** antes de la clase.

---

## 1. Teoría (15 min, antes del notebook)

| Min | Tema | Idea para la diapositiva |
|---|---|---|
| 0–3 | Repaso del Taller 1 | Filtros, Viola-Jones y tareas de YOLO; hoy: abrir la caja negra |
| 3–6 | Cómo funciona YOLO | Columna (filtros aprendidos) → cuello (tamaños) → cabeza (cajas, clase, confianza); una sola pasada |
| 6–9 | Transferencia de aprendizaje | No se entrena desde cero: se reutiliza lo que el modelo ya sabe ver y se le enseñan clases nuevas |
| 9–12 | Cómo se mide un detector | Precisión (pocas falsas alarmas), recall (pocos objetos perdidos), mAP (calidad general) |
| 12–15 | De detectar a entender rostros | Detectar → puntos clave → alinear → clasificar (expresión) o identificar (proyecto final) |

---

## 2. Práctica (45 min): sección por sección

### Resumen de tiempos

| Min | Sección | Interactivo |
|---|---|---|
| 0–2 | 0. Configuración | — |
| 2–4 | 1. Una sola pasada | — |
| 4–7 | 2. Confianza y cajas repetidas | Sí |
| 7–9 | 3. Límites del modelo de fábrica | — |
| 9–13 | 4. OpenCV antes y después | Sí |
| 13–15 | 5. Dataset propio | — |
| 15–16 | 6. Lanzar el entrenamiento en segundo plano | — |
| 16–25 | 6. **Mientras entrena:** hiperparámetros, aumento de datos, predicción, archivo YAML, progreso | Sí |
| 25–29 | 7. Evaluación | — |
| 29–31 | 8. Antes vs después + exportar a ONNX | — |
| 31–34 | 9. Rostros con puntos clave | — |
| 34–38 | 10. Expresiones faciales | — |
| 38–42 | 11. Prueba con tu cámara | Sí |
| 42–45 | Cierre | — |

**Clave de tiempo:** el entrenamiento corre **en segundo plano**, así que la clase no se detiene a esperar: mientras entrena se trabaja la configuración de Ultralytics. Si el grupo va lento, lanza tú las Secciones 5 y 6 al inicio. **No recortes** las Secciones 6, 7 y 10.

---

### Sección 0. Configuración (min 0–2)

**Antes de clase (enviar a los estudiantes):**
- Cuenta de Roboflow.
- Secreto `ROBOFLOW_API_KEY` creado en Colab, con *Acceso al notebook* activado.
- Entorno con **GPU T4**.

**Qué decir:**
- **Ejecutar todas** apenas entren.
- Si la sesión se reinicia sola al instalar, es normal (corrección de OpenCV 5, igual que en el Taller 1): volver a pulsar **Ejecutar todas**.

**Qué deben ver:**
- "Instalación lista."
- El nombre de la GPU (Tesla T4).
- "Material listo".

---

### Sección 1. Una sola pasada (min 2–4)

**Qué explicar:**
- YOLO devuelve, para cada objeto: **clase**, **confianza** y **caja** (esquinas x1, y1, x2, y2).
- La tabla muestra 5 objetos en la foto del bus: 4 personas y el bus.
- Las dos imágenes son iguales: una dibujada **con OpenCV a mano** y otra con `r.plot()`. Mensaje: **los resultados de YOLO son solo números**; con OpenCV hacemos lo que queramos con ellos.

---

### Sección 2. Confianza y cajas repetidas (min 4–7)

**Qué explicar:**
- La red propone muchas cajas; la **confianza mínima** filtra las dudosas.
- YOLO11 usa **NMS** para borrar cajas repetidas sobre el mismo objeto. YOLO26 aprendió a no repetirlas y **no necesita NMS**.

**Qué valores mover** (número de cajas: YOLO11 / YOLO26):

| Confianza | Solapamiento 0.3 | Solapamiento 0.7 (inicial) | Solapamiento 0.95 | Qué decir |
|---|---|---|---|---|
| 0.25 (inicial) | 5 / 5 | 5 / 5 | **8 / 5** | Con solapamiento 0.95 YOLO11 empieza a repetir |
| **0.05** | 10 / 10 | 11 / 10 | **20 / 10** | **El ejemplo más claro:** YOLO11 duplica todo, YOLO26 no cambia |
| 0.5 | 5 / 5 | 5 / 5 | 6 / 5 | Con confianza alta hay menos cajas candidatas |

**Pregunta:** "¿Por qué a YOLO26 no le afecta el solapamiento?" Porque ya no usa NMS: aprendió a dar una sola caja por objeto.

---

### Sección 3. Límites del modelo de fábrica (min 7–9)

**Qué deben ver:**

| Escena | Predicción de `yolo26n.pt` |
|---|---|
| Autos vistos desde arriba | *cell phone*, *cell phone* |
| Limón | *sports ball* |
| Piña | *vase* |
| Personas | *person* ×3 (no existe la clase rostro) |

**Mensaje:** el modelo solo conoce lo que aprendió. Soluciones: **entrenar con datos propios** (Parte 3) o usar un **modelo especializado** (Parte 4).

---

### Sección 4. OpenCV antes y después del modelo (min 9–13)

**Antes del modelo (control de ruido y filtro).** Objetos detectados; "bus" indica si el bus sigue apareciendo:

| Ruido | Ninguno | Gaussiano | Mediana |
|---|---|---|---|
| 0 % | 5 (con bus) | 5 (con bus) | 5 (con bus) |
| 10 % | 3 (**sin bus**) | 4 (con bus) | 4 (con bus) |
| 20 % | 3 (sin bus) | 3 (sin bus) | 4 (**con bus**) |
| **30 % (inicial)** | 3 (**sin bus**) | 4 (sin bus) | 4 (**con bus**) |
| 40 % | 4 (sin bus) | 4 (sin bus) | 5 (con bus) |

**Qué decir:** el mismo modelo, con la misma imagen, gana o pierde el bus según el **filtro aplicado antes**. Es la regla del Taller 1 ("primero limpiar"), ahora aplicada a una red neuronal.

**Dato extra para la discusión:** en pruebas aparte, oscurecer la imagen no afectó a YOLO26 y mejorar el contraste (CLAHE) no lo ayudó. El modelo aprendió con muchas variaciones de brillo. **No todo preprocesamiento ayuda: hay que medir.**

**Después del modelo:**
- Con las cajas, OpenCV **recorta** cada persona (4 recortes) y las **pixela**.
- Aplicaciones: privacidad (difuminar rostros o placas) y pasar cada recorte a otro modelo, como en la Sección 10.

---

### Sección 5. Dataset propio con Roboflow (min 13–15)

**Qué explicar:**
- **Roboflow:** anotar, aumentar datos y exportar.
- El formato YOLO es una línea por objeto: clase y caja normalizada.
- Los conjuntos de **entrenamiento**, **validación** y **prueba** cumplen roles distintos. La prueba es como el examen final: el modelo nunca la ve.
- Es el mismo dataset de frutas del taller con YOLO11.

**Qué deben ver:** el contenido de `data.yaml`, el número de imágenes por conjunto y 4 imágenes con sus cajas dibujadas.

---

### Sección 6. Fine-tuning en segundo plano y configuración de Ultralytics (min 15–25)

**Lanzar (1 min).**
- La celda imprime y ejecuta el comando `yolo detect train ...`: el mismo de la terminal del taller con YOLO11.
- Corre como **proceso en segundo plano**, así que el notebook queda libre.
- **Qué explicar:** transferencia de aprendizaje (se parte de `yolo26n.pt`, por eso bastan pocas épocas). Se entrena con **15 épocas**. **Tiempo medido en Colab con T4: unos 0,3–0,4 min por época → 5–6 min en total.** La celda de seguimiento muestra el tiempo real de cada época. Para un proyecto real se usan más épocas, con `patience` para detenerse a tiempo.

**Mientras entrena (9 min), en este orden:**

1. **Tabla de parámetros.** Explica que todo se configura igual en Python (`model.train(lr0=...)`) y en la terminal (`yolo train lr0=...`). Destaca cinco:
   - `epochs` / `patience`: cuánto entrenar y cuándo parar.
   - `imgsz`: tamaño de imagen (objetos pequeños vs velocidad).
   - `lr0`: tamaño de los ajustes.
   - `freeze`: congelar capas con pocos datos.
   - Los de **aumento de datos**.

2. **Aumento de datos (interactivo).** Son operaciones de OpenCV del Taller 1, aplicadas al azar en cada época. Las cajas se mueven con la imagen: también se transforman.
   - Valores iniciales: saturación +0.5, brillo −0.3, rotación 10°, zoom 0.2, desplazamiento 0.1, espejo.
   - Mover **rotación a 45°**: las cajas se deforman y aparecen bordes grises. Por eso los valores por defecto son moderados (`degrees=0`).
   - Activar **mosaic**: 4 imágenes en una. Ayuda con objetos pequeños y está activo por defecto (`mosaic=1.0`).
   - Ejecutar la celda de `train_batch0.jpg`: es lo que **realmente** vio el modelo en su primer lote.

3. **Parámetros de predicción (interactivo, frame 244 del pasillo).** Detecciones y tiempo en CPU según `imgsz`:

   | imgsz | Personas detectadas (hay 2) | Tiempo (CPU de prueba) |
   |---|---|---|
   | 160 | **0** | ~6 ms |
   | 320 | 2 | ~9 ms |
   | 640 (inicial) | 2 | ~23 ms |
   | 1280 | 2 | ~75 ms |

   - Mensaje: **compromiso entre detectar objetos pequeños y velocidad**.
   - `max_det=1` deja solo un objeto.
   - `classes=[0]` filtra personas.
   - **Ojo:** Ultralytics **recuerda** `classes` entre llamadas. Por eso esta celda usa una instancia aparte del modelo.

4. **Archivo YAML.** Guardar la configuración (`output/mi_config.yaml`) y reutilizarla con `cfg=`. Mencionar también:
   - `resume=True`: continuar un entrenamiento interrumpido.
   - `freeze=10`: congelar las primeras 10 capas.
   - `model.tune()`: búsqueda automática de hiperparámetros. Es muy costosa (cientos de entrenamientos); **no se ejecuta en clase**.

5. **Progreso.** Ejecutar `progreso()` varias veces. La pérdida al ubicar cajas debe bajar y el mAP subir. Si la pérdida de validación sube mientras la de entrenamiento baja, el modelo está memorizando (*overfitting*).

**Seguimiento época a época:** la celda imprime una fila por cada época terminada:

| época | min | box_loss | cls_loss | precisión | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|

- Antes de entrenar, el notebook **borra la carpeta del entrenamiento anterior** (`runs/frutas`), para que no se mezclen resultados viejos en el seguimiento.
- **Qué señalar mientras aparecen las filas:** las pérdidas (`box_loss`, `cls_loss`) **bajan** y las métricas **suben**. La columna `min` muestra cuánto tarda cada época.
- Al terminar imprime el tiempo total y el gráfico de progreso. Si el entrenamiento falla, muestra el final del registro (`output/entrenamiento.log`).

**Plan B:** sube antes de clase un `best.pt` a tu Drive y usa la celda del Plan B si falla la GPU o la sesión.

---

### Sección 7. Evaluación (min 25–29)

**Qué explicar (sin fórmulas), apoyándote en el ejemplo del notebook:** hay 10 naranjas reales; el modelo dibuja 8 cajas, 6 correctas y 2 errores, y se le escapan 4.

| Métrica | Idea | En el ejemplo |
|---|---|---|
| **IoU** | Cuánto coincide la caja predicha con la real; con IoU ≥ 0.5 cuenta como acierto | — |
| **Precisión** | De lo detectado, ¿cuánto era correcto? | 6 de 8 = 75 % |
| **Recall** | De lo que había, ¿cuánto encontró? | 6 de 10 = 60 % |
| **F1** | Equilibrio entre ambas | ≈ 67 % |
| **AP** | Calidad de una clase considerando todos los umbrales (área bajo la curva precisión-recall) | Curva del notebook |
| **mAP50 / mAP50-95** | Promedio del AP de todas las clases; el 50-95 exige cajas más precisas | Tabla del notebook |
| ***Accuracy*** | Aciertos sobre el total: se usa en **clasificación**, no en detección | — |

- **¿Por qué no *accuracy* en detección?** Porque el "fondo vacío correctamente" es infinito y no se puede contar. En clasificación (por ejemplo `yolo26n-cls` o el modelo de expresiones) sí se usa.
- **Precisión vs recall:** al subir la confianza, sube la precisión y baja el recall. La curva F1 muestra el umbral más equilibrado.
- **Matriz de confusión:** la columna *background* son objetos no encontrados; la fila *background*, falsas alarmas.

**Qué anotar en tu ensayo:**
- Precisión, recall, mAP50 y mAP50-95 globales (el notebook los imprime en una línea).
- La clase con peor resultado y si su problema es de precisión o de recall.

**Referencia:** en el taller con YOLO11s (10 épocas) se obtuvo mAP50 ≈ 0.73 y mAP50-95 ≈ 0.44; *Orange* tuvo el recall más bajo (≈ 0.50).

**Pregunta:** "¿Qué harían para mejorar la clase con peor resultado?" Más imágenes de esa clase, mejores anotaciones o más épocas.

---

### Sección 8. Antes vs después y exportación (min 29–31)

**Qué decir:**
- El modelo entrenado ahora reconoce **las clases del dataset**.
- Si una fruta no estaba en el entrenamiento, la confunde con la más parecida: **los datos definen lo que el modelo puede ver**.

**Exportar:**
- `best.export(format='onnx')` convierte el modelo a un formato portable, que se puede usar sin PyTorch (incluso con `cv2.dnn`).
- Otros destinos: TensorRT (GPU NVIDIA), OpenVINO (Intel) y TFLite (celulares).
- La celda compara la velocidad en CPU. En la prueba, ONNX fue algo más rápido que PyTorch (7 vs 8 ms con el modelo de prueba); en Colab los tiempos serán otros.

---

### Sección 9. Rostros con puntos clave (min 31–34)

**Qué deben ver** (Viola-Jones del Taller 1 vs YOLO26n-face):

| Frame | Escena | OpenCV (Viola-Jones) | YOLO26n-face |
|---|---|---|---|
| 244 | 2 rostros pequeños | 0 | **2** |
| 305 | Perfil y movido | 0 | **1** |
| 488 | 3 frontales | 3 | 3 |

**Qué explicar:**
- YOLO26n-face fue entrenado por la comunidad con miles de rostros (*fine-tuning*, igual que hicimos con frutas).
- Entrega **5 puntos clave**: ojos, nariz y comisuras de la boca.

---

### Sección 10. Expresiones faciales (min 34–38)

**Qué explicar (la tabla de 3 pasos):**
1. **YOLO26n-face:** rostro y 5 puntos clave.
2. **OpenCV:** usa los puntos clave para **enderezar** el rostro a 112×112, con los ojos y la boca siempre en el mismo lugar. Es la misma idea de las transformaciones del Taller 1.
3. **Modelo de expresiones** (MobileFaceNet, OpenCV Zoo, ejecutado con `cv2.dnn`): 7 expresiones.

**Qué deben ver:**
- **Frame 900:** 2 rostros, ambos "felicidad" al 100 %.
- Recortes alineados y gráfico de barras de probabilidades.
- **GIF:** las dos personas pasan de neutral → sorpresa → felicidad → neutral (frames 830 a 970).

**Pregunta:** "¿Por qué alinear antes de clasificar?" El modelo aprendió con rostros alineados; sin alinear, una cabeza inclinada confunde al modelo.

---

### Sección 11. Prueba con tu cámara (min 38–42)

**Qué hacer:**
- Cada estudiante ejecuta la celda, permite la cámara y pulsa **Capturar**.
- Pedir que prueben **sonreír, sorprenderse, cara seria y triste**.
- Sin cámara: cambiar `USAR_CAMARA = False` y subir una foto.
- Fuera de Colab, la celda usa el frame 858 del video (2 rostros con "sorpresa" al 100 %).

**Qué esperar:**
- Felicidad, sorpresa y neutral se reconocen muy bien.
- Tristeza, enojo, asco y miedo son más difíciles y a veces se confunden entre sí. Úsalo para discutir.

**Discusión ética (1 min):**
- Una **expresión no es una emoción**.
- Hay sesgos según las personas y las condiciones con que se entrenó el modelo.
- El **AI Act** de la UE prohíbe el reconocimiento de emociones en trabajo y educación (salvo razones médicas o de seguridad).
- La **LOPDP** de Ecuador trata los datos biométricos como sensibles.

---

### Cierre (min 42–45)

**Qué decir:**
- Repasar la cadena **OpenCV → YOLO → OpenCV → clasificador**.
- **Proyecto final:** la misma cadena con un modelo de **reconocimiento** en lugar del de expresiones (por ejemplo SFace, de OpenCV Zoo).

**Reto:**
1. Combinación con más cajas repetidas en YOLO11 (respuesta: confianza 0.05 y solapamiento 0.95, 20 cajas).
2. Ruido desde el que se pierde el bus sin filtro (respuesta: 10 %).
3. Entrenar con otro dataset de Roboflow.
4. Reentrenar con `output/mi_config.yaml` y comparar el mAP.
5. Expresiones en video con `vid_stride`.

---

## 3. Preguntas probables (con respuesta)

- **¿Por qué YOLO26 no necesita NMS?** Fue entrenado para dar una sola caja por objeto. Eso simplifica el modelo y lo hace más rápido en dispositivos pequeños.
- **¿Cuántas imágenes necesito para entrenar?** No hay un número fijo. Como referencia, unos cientos de ejemplos por clase y variedad de condiciones (luz, ángulos, fondos). Importa más la variedad que la cantidad.
- **¿Por qué evaluar con el conjunto de prueba?** El "mejor modelo" se eligió mirando la validación; la prueba es una medición honesta con imágenes nunca vistas.
- **¿Puedo entrenar sin GPU?** Sí, pero muy lento. El notebook usa un modo reducido en CPU solo para ver el proceso.
- **¿El modelo de expresiones también se entrenó con *fine-tuning*?** Es una red MobileFaceNet entrenada con el dataset RAF-DB (fotos reales de expresiones). Se podría ajustar con fotos propias, como hicimos con las frutas.
- **¿Puedo usar esto para vigilar a mis estudiantes o empleados?** En la UE está prohibido en educación y trabajo, y en Ecuador requiere base legal y consentimiento. Es un ejemplo didáctico.
- **¿Qué cambia para el proyecto final?** Solo el último paso: en vez de "¿qué expresión tiene?", "¿quién es?" (un modelo de reconocimiento facial que compara cada rostro con una galería de personas registradas).
- **¿Qué hiperparámetros conviene tocar primero?** `epochs` y `patience`, luego `imgsz` (objetos pequeños) y el aumento de datos según tu escena (por ejemplo `flipud` solo tiene sentido en imágenes aéreas). `lr0` y `optimizer` rara vez hace falta cambiarlos: `optimizer=auto` elige bien.
- **¿Por qué entrenar en segundo plano?** Para no bloquear el notebook. En un proyecto real es lo normal: el entrenamiento corre en una terminal o servidor y se revisa su progreso en `results.csv`.
- **¿`model.tune()` siempre mejora?** Suele mejorar unos puntos de mAP, pero cuesta muchas horas de GPU. Primero hay que mejorar los datos.

---

## 4. Riesgos técnicos y plan B

| Riesgo | Plan B |
|---|---|
| No hay GPU T4 disponible | El notebook entrena en modo reducido en CPU (3 épocas, 25 % de los datos). Para la evaluación, usar el `best.pt` del Plan B. |
| El secreto de Roboflow no funciona | El código pide la clave a mano (`getpass`). Verificar *Acceso al notebook* en el panel de Secretos. |
| La sesión se reinicia al instalar | Esperado (corrección de OpenCV 5): volver a **Ejecutar todas**. |
| El entrenamiento tarda más de lo previsto | Bajar `epochs` a 10 en `CFG`, o usar el Plan B. |
| El entrenamiento en segundo plano falla | La celda de espera muestra el final de `output/entrenamiento.log`. Error típico: falta la clave de Roboflow o la ruta de `data.yaml`. |
| Las celdas interactivas van lentas mientras entrena | Es normal: comparten la máquina con el entrenamiento. Usa valores de `imgsz` pequeños. |
| La cámara no abre | Revisar los permisos del navegador; si no, `USAR_CAMARA = False` y subir una foto. |
| Un estudiante no quiere usar su cámara | Usar el frame de ejemplo o una foto que no sea de una persona real del grupo. |
| URLs externas caídas (modelos o videos) | Tener la carpeta `material/` en tu Drive y copiarla; el código no descarga lo que ya existe. |

---

## 5. Checklist antes de clase
- [ ] Ensayar el notebook completo en Colab gratuito con T4; anotar el tiempo de entrenamiento y las métricas.
- [ ] Subir `best.pt` a Drive (Plan B).
- [ ] Enviar a los estudiantes: cuenta de Roboflow + secreto `ROBOFLOW_API_KEY` + GPU T4.
- [ ] Probar la cámara en el aula (navegador y permisos).
- [ ] Repasar las tablas de las Secciones 2, 4, 9 y 10.

## 6. Material de apoyo
- [Ultralytics: entrenamiento](https://docs.ultralytics.com/modes/train/) · [Métricas de YOLO](https://docs.ultralytics.com/guides/yolo-performance-metrics/) · [YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [YOLO-face (akanametov)](https://github.com/akanametov/yolo-face) · [Expresiones faciales, OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition) · [SFace, OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
- [Roboflow Universe](https://universe.roboflow.com/)
