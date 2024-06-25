import random
import tkinter as tk
from tkinter import messagebox
from sympy import symbols, Matrix, diff
x_star = symbols('x*', real=True)
nabla_f = symbols('∇f', real=True)
H_f = symbols('∇²f', real=True)
x1, x2, f = symbols('x1 x2 f', real=True)
f_x1x1 = diff(f, x1, x1)
f_x2x2 = diff(f, x2, x2)
f_x1x2 = diff(f, x1, x2)

preguntas = {
    "P01": ("El metodo de dos faces se aplica unicamente en problemas de minimización", False),
    "P02": ("Al incrementar/reducir un recurso, se puede asegurar que el valor de la función objetivo no"
            "cambiará si el cambio total en la función objetivo dividido por el cambio en el recurso es igual al precio sombra.", False),
    "P03": ("El precio sombra es la sensibilidad a cambios en los lados derechos de las restricciones.", True),
    "P04": ("El método de dos fases se aplica cuando hay restricciones de tipo ""mayor o igual que.", True),
    "P05": ("Si una restricción tiene como lado derecho un valor negativo, es suficiente con multiplicarla por -1 para que pueda ser usada en el método simplex sin violar las condiciones de no negatividad.", False),
    "P06": ("En un modelo de programación lineal, la función objetivo es una recta o plano que tiene el mismo valor para cualquier punto en el que se evalúe.", True),
    "P07": ("Al incrementar/reducir un recurso, se puede determinar que las variables básicas no cambiarán si el cambio total en la función objetivo dividido por el cambio en el recurso es igual al precio sombra.", True),
    "P08": ("Al resolver un problema de maximización de programación lineal por el método gráfico, siendo negativos todos los coeficientes de la función objetivo, la solución se encontrará en el punto más alejado al origen del sistema de coordenadas.", False),
    "P09": ("Un precio sombra positivo en un tablero final del método simplex implica que al hacer crecer el lado derecho de la restricción, el valor de la función objetivo siempre crecerá.", False),
    "P10": ("El precio sombra es la sensibilidad del modelo a cambios en los parámetros de la función objetivo", False),
    "P11": ("Las variables artificiales son las variables básicas de inicio en las restricciones de tipo ""mayor o igual que""en la primera fase del método de dos fases.", True),
    "P12": ("El precio sombra solo se puede observar en restricciones saturadas.", True),
    "P13": ("El precio sombra es el incremento del valor de la función objetivo al agregar variables de holgura o exceso", False),
    "P14": ("En un modelo de programación lineal, la función objetivo es el producto vectorial del vector de variables de decisión y el vector de coeficientes.", True),
    "P15": ("En la forma definida en clases, el método simplex itera hasta que no encuentra valores positivos en los coeficientes de la función objetivo.", False),
    "P16": ("El método de dos fases obligatoriamente tiene cero como valor final de la función objetivo en la segunda fase.", False),
    "P17": ("El precio sombra es la derivada de una restricción saturada con respecto a la función objetivo.", False),
    "P18": ("Al incrementar/reducir un recurso, se puede asegurar que el valor de la función objetivo no cambiará si el cambio total en la función objetivo dividido por el cambio en el recurso es igual al precio sombra.", False),
    "P19": ("El precio sombra es la derivada de la función objetivo con respecto a una restricción.", True),
    "P20": ("Las variables artificiales se agregan para poder tener coeficientes negativos en la función objetivo en las restricciones de tipo -mayor o igual que- en la primera fase del modelo de dos fases.", True),
    "P21": ("Al resolver un problema de maximización de programación lineal por el método gráfico, siendo positivos todos los coeficientes de la función objetivo y teniendo dos restricciones de tipo mayor o igual que, el problema es factible.", True),
    "P22": ("Al resolver un problema de programación no lineal por el método de fuerza bruta, la complejidad del problema depende del número de volúmenes en que se subdivide la región factible.", True),
    "P24": ("La diferencia entre el método básico del gradiente descendente y el de actualización de f por el método de Barzilai- Borwein radica en la definición inicial del paso de salto Ax.", False),
    "P25": ("Las variables de holgura se agregan cuando hay restricciones de tipo 'menor o igual que'.", True),
    "P26": ("Las variables artificiales se eliminan del problema en la segunda fase.", True),
    "P27": ("Una matriz es semidefinida negativa aunque alguno de sus valores propios sea mayor a cero.", False),
    "P28": ("Una solución a* de un problema de programación no lineal se define como un máximo global si f(x*) ≥ f(xƒ), Væ Є R2", True),
    "P29": ("El método simplex garantiza que la solución óptima se encuentra en un vértice de la región factible.", True),
    "P30": ("En un modelo de programación lineal, la función objetivo es la combinación lineal de los recursos.", False),
    "P31": ("Las condiciones suficientes de segundo orden permiten determinar cuando un punto es un mínimo o máximo local estricto de una función.", True),
    "P32": ("La validación es el proceso de prueba y depuración del modelo matemático.", True),
    "P33": ("f""El criterio H(x) = ∂²f(x)/∂x1² * ∂²f(x)/∂x2² - (∂²f(x)/∂x1∂x2)² para matrices hessianas 2x2 solo puede utilizarse si estas matrices son simétricas.", True),
    "P34": ("Una matriz es semidefinida positiva si y solo si todos su valores propios son mayores o iguales a cero.", True),
    "P35": ("En la forma definida en clases, el método simplex itera hasta que no encuentra valores positivos en los coeficientes de la función objetivo..", False),
    "P36": ("En un un problema generalizado de optimización se busca optimizar cualquier función de las variables de decisión establecida.", True),
    "P37": ("Al resolver un problema de programación no lineal por el método de fuerza bruta, la precisión del problema no depende del número de volúmenes en que se subdivide la región factible.", False),
    "P38": ("La interpretación del precio sombra impone al 'propietario' del problema la cuestión sobre el incremento/reducción de uno de los recursos, pero no expone el costo de realizar este incremento.", True),
    "P39": ("La diferencia principal entre el algoritmo del gradiente descendente y el método de Newton es la utilización de la matriz hessiana en este último para determinar la dirección de búsqueda del punto óptimo.", True),
    "P40": ("Debido a las condiciones suficientes de segundo orden si la matriz hessiana se anula en el punto evaluado no es posible determinar si es un mínimo o máximo local estricto.", False),
    "P41": ("Las variables de holgura se agregan en la función objetivo durante la primera fase.", False),
    "P42": ("Suponga que {H_f}(x) es continua en V_{x_star}, que {nabla_f}(x_star) = 0 y que {H_f}(x_star) es definida negativa, entonces x* es un mínimo local estricto de f(x*).", False),
    "P43": ("El valor de la función objetivo de un problema de programación entera de maximización es menor o igual al de la relajación continua.", True),
    "P44": ("Los algoritmos genéticos paralelos de grano fino permiten el cruce entre individuos que pertenecen a vecindarios no adyacentes.", False),
    "P46": ("La actualización de f en el método de Barzilai-Borwein tiene como objetivo acelerar la convergencia a un punto crítico local.", True),
    "P47": ("En un problema de programación no lineal tanto la función objetivo como las restricciones deben ser no lineales.", False),
    "P48": ("La medida de aptitud es el equivalente de la función objetivo en algoritmos genéticos.", True),
    "P49": ("Debido a las condiciones suficientes de segundo orden si la matriz hessiana se anula en el punto evaluado no es posible determinar si es un mínimo o máximo local estricto.", False),
    "P50": ("Las variables de holgura se agregan cuando hay restricciones de tipo 'mayor o igual que'", False),
    "P51": ("El criterio {criterio_Hx} para matrices hessianas 2x2 es incapaz de dar información sobre un punto crítico si la función es simétrica con respecto a las variables en el mismo, es decir, la forma de la función es igual para ambas variables.", True),
    "P52": ("Un punto óptimo local se define como el punto x*, si en Væ Є Væ*, x* es el único punto óptimo local.", False),
    "P54": ("La diferencia principal entre el algoritmo del gradiente descendente y el método de Newton es la utilización de la matriz hessiana en este último para determinar la dirección de búsqueda del punto óptimo.", True),
    "P55": ("Cuando en el método de Newton se evalúa ||Vx_n+1|| < ||Vx_n||, se establece, implícitamente, que lo que se busca es un máximo.", False),
    "P56": ("En el algoritmo de colonia de abejas las abejas espectadoras determinan probabilísticamente la fuente a explorar.", True),
    "P58": ("Las condiciones necesarias de segundo orden se usan para determinar cuando el gradiente se anula en el óptimo y, a la vez, cuando la matriz hessiana es semidefinida positiva o negativa (dependiendo de si es un mínimo o un máximo).", True),
    "P59": ("En el algoritmo de recocido el estado de energía más alto representa una mejor solución.", False),
    "P60": ("A partir de dos padres, un algoritmo genético siempre generará dos hijos, dada la regla de cruzamiento.", True),
    "P61": ("En un algoritmo de colonias de hormigas las feromonas se evaporan para evitar llegar a óptimos locales.", True),
    "P62": ("La derivada numérica de una función es más precisa mientras más pequeño sea el valor de Ax, sin importar que tan pequeño sea el mismo.", False),
    "P63": ("El método simplex no es un algoritmo greedy (codicioso).", False),
    "P64": ("En un algoritmo de colonias de hormigas la deposición de feromonas es directamente proporcional al costo de la solución.", False),
    "P65": ("Supongamos que el coeficiente más negativo de la primera fila en una iteración del método simplex para un problema de maximización corresponde a una variable de holgura o de exceso, y la fila pivote corresponde a una variable de decisión. Por tanto, el debería sacarse del grupo de variables básicas a la variable de decisión e introducir al mismo la respectiva variable de holgura/exceso. No obstante, no debe hacerse este cambio pues la función objetivo disminuiría en valor.", False),
    "P66": ("Las variables artificiales son las variables básicas de inicio en las restricciones de tipo 'mayor o igual que' en la primera fase del método de dos fases.", True),
    "P67": ("El proceso de estudio de la solución obtenida, evaluando la flexibilidad de la misma con respecto a los parámetros suministrados al modelo matemático se denomina validación de la solución.", False),
    "P68": ("La interpretación del precio sombra impone al 'propietario' del problema la cuestión sobre el incremento/reducción de uno de los recursos, pero no expone el costo de realizar este incremento.", True),
    "P69": ("El método de dos fases se aplica únicamente en problemas de minimización.", False),
    "P70": ("El método de dos fases obligatoriamente tiene cero como valor final de la función objetivo en la primera fase.", True),
    "P71": ("El precio sombra es la utilidad marginal de un recurso.", True),
    "P72": ("Si una restricción tiene como lado derecho un valor negativo y todos los coeficientes del lado izquierdo son negativos, es suficiente con multiplicarla por -1 para que pueda ser usada en el método simplex sin violar las condiciones de no negatividad.", True),
    "P73": ("El precio sombra es el incremento del valor de la función objetivo al modificar los coeficientes de la misma.", False),
    "P74": ("Las variables artificiales se agregan para poder tener coeficientes negativos en la función objetivo en las restricciones de tipo 'mayor o igual que' en la primera fase del modelo de dos fases.", True),
    "P75": ("La fila pivote se selecciona a partir del menor valor que se obtiene al dividir el lado derecho de las restricciones por los coeficientes de la columna pivote.", False),
    "P76": ("Una restricción de tipo 'menor o igual que' es redundante en un problema de minimización.", False),
    "P76": ("El precio sombra de las variables básicas se encuentra en la primera fila de un tablero simplex final.", False),
    "P76": ("Una variable de holgura o exceso que tiene valor igual a cero tiene un precio sombra.", True),
    "P76": ("Dadas las restricciones de no negatividad del modelo, ningún coeficiente del lado izquierdo de las restricciones puede ser negativo.", False),
    "P76": ("Para probar que una función tiene un óptimo global solo es necesario probar que ésta es convexa.", False),
}

preguntas_seleccionadas = random.sample(list(preguntas.keys()), 30)

preguntas_seleccionadas_dict = {pregunta_id: preguntas[pregunta_id] for pregunta_id in preguntas_seleccionadas}

def presentar_pregunta(num_pregunta):
    pregunta_id = next(iter(preguntas_seleccionadas_dict.keys()))
    pregunta_texto, respuesta_correcta = preguntas_seleccionadas_dict.pop(pregunta_id)
    
    def verificar_respuesta(respuesta_usuario, nivel_certezas_var):
        nivel_certezas = nivel_certezas_var.get()
        if respuesta_usuario == respuesta_correcta:
            if nivel_certezas == "Bajo":
                puntaje = 1
            elif nivel_certezas == "Medio":
                puntaje = 2
            else:  # Alto
                puntaje = 3
            messagebox.showinfo("Respuesta", f"¡Respuesta correcta! Puntaje: +{puntaje}")
            sumar_puntaje(puntaje)
        else:
            if nivel_certezas == "Bajo":
                puntaje = -1
            elif nivel_certezas == "Medio":
                puntaje = -3
            else:  # Alto
                puntaje = -6
            messagebox.showinfo("Respuesta", f"Respuesta incorrecta. La respuesta correcta es {'Verdadero' if respuesta_correcta else 'Falso'}. Puntaje: {puntaje}")
            sumar_puntaje(puntaje)
        
        if preguntas_seleccionadas_dict:
            presentar_pregunta(num_pregunta + 1)
        else:
            messagebox.showinfo("Fin del juego", f"¡Has completado todas las preguntas! Puntaje final: {puntaje_total}")
            ventana.quit()

    label_pregunta.config(text=f"{num_pregunta}.- {pregunta_texto}")
    
    certeza_label.pack()
    certeza_combo.pack()
    
    boton_verdadero.config(command=lambda: verificar_respuesta(True, nivel_certezas))
    boton_falso.config(command=lambda: verificar_respuesta(False, nivel_certezas))

def sumar_puntaje(puntos):
    global puntaje_total
    puntaje_total += puntos
    label_puntaje.config(text=f"Puntaje: {puntaje_total}")

ventana = tk.Tk()
ventana.title("Verdadero o Falso")

label_pregunta = tk.Label(ventana, text="", font=("Helvetica", 30), wraplength=1000)
label_pregunta.pack()

certezas = ["Bajo", "Medio", "Alto"]
certeza_label = tk.Label(ventana, text="Nivel de Certeza:", font=("Helvetica", 30))
nivel_certezas = tk.StringVar()
nivel_certezas.set(certezas[0])
certeza_combo = tk.OptionMenu(ventana, nivel_certezas, *certezas)

frame_botones = tk.Frame(ventana)
frame_botones.pack()

boton_verdadero = tk.Button(frame_botones, text="Verdadero", width=10, height=2, font=("Helvetica", 30))
boton_verdadero.grid(row=0, column=0, padx=10, pady=10)

boton_falso = tk.Button(frame_botones, text="Falso", width=10, height=2, font=("Helvetica", 30))
boton_falso.grid(row=0, column=1, padx=10, pady=10)

label_puntaje = tk.Label(ventana, text="Puntaje: 0", font=("Helvetica", 30))
label_puntaje.pack()

puntaje_total = 0

presentar_pregunta(1)

ventana.mainloop()
