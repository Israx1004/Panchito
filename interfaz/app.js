(() => {
  const $ = (sel) => document.querySelector(sel);
  const log = $("#log");
  const form = $("#formulario");
  const caja = $("#pregunta");
  const botonEnviar = $("#enviar");
  const botonNueva = $("#nueva");
  const ejemplos = document.querySelectorAll(".ejemplo");
  const estadoEl = $("#estado");
  const estadoTexto = $("#estado-texto");
  const bienvenida = log.innerHTML;
  const listaTips = $("#tips");
  const botonTip = $("#otro-tip");
  let ocupado = false;
  let glucosasPrevias = 0;

  const el = (tag, clase, texto) => {
    const n = document.createElement(tag);
    if (clase) n.className = clase;
    if (texto !== undefined) n.textContent = texto;
    return n;
  };
  const icono = (id, clase = "icono") => {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("class", clase);
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(ns, "use");
    use.setAttribute("href", `#${id}`);
    svg.append(use);
    return svg;
  };
  const num = (v, dec = 1) => Number(v).toLocaleString("es-MX", { maximumFractionDigits: dec });

  //consejos
  const TIPS = [
    ["El método del plato.", "La mitad con verduras, un cuarto con proteína magra y un cuarto con carbohidratos, de preferencia integrales."],
    ["Fibra todos los días.", "Se recomiendan de 25 a 30 g al día: frijol, lenteja, avena, verduras y fruta entera. Auméntala poco a poco y toma agua."],
    ["Fruta entera, no jugo.", "El jugo concentra el azúcar y pierde la fibra que hace que la glucosa suba más lento."],
    ["Agua natural primero.", "Limita refrescos, jugos y bebidas azucaradas: suben la glucosa en minutos."],
    ["Carga glucémica.", "Combina el índice glucémico con los carbohidratos de la porción: 10 o menos es baja, de 11 a 19 media, 20 o más alta."],
    ["Anota tus mediciones.", "Valor, hora y lo que comiste: es lo que más ayuda en tu consulta. Cuéntamelas y las anoto en tu monitoreo."],
    ["Tortilla de maíz.", "Tiene más fibra y menos grasa que la de harina. Igual cuenta la porción: cada tortilla suma carbohidratos."],
  ];
  const VISIBLES = innerHeight >= 1000 ? 3 : 2;   // en pantallas bajas caben 2 sin tapar el monitoreo
  let siguienteTip = 0;

  function globoTip([titulo, texto], animar) {
    const li = el("li", animar ? "globo entra" : "globo");
    li.append(el("b", "", titulo), ` ${texto}`);
    return li;
  }
  function pintarTips() {
    listaTips.replaceChildren();
    for (let i = 0; i < VISIBLES; i++) listaTips.append(globoTip(TIPS[(siguienteTip + i) % TIPS.length], false));
    siguienteTip = VISIBLES % TIPS.length;
  }
  botonTip.addEventListener("click", () => {
    listaTips.firstElementChild?.remove();
    listaTips.append(globoTip(TIPS[siguienteTip], true));
    siguienteTip = (siguienteTip + 1) % TIPS.length;
  });

  //estado de conexion
  function pintarEstado(tipo, texto) {
    estadoEl.dataset.estado = tipo;
    estadoTexto.textContent = texto;
  }

  async function revisarEstado() {
    if (ocupado) return;
    const e = await Asistente.estado();
    if (e.servidor === false) pintarEstado("servidor no conectado", "Servidor no conectado");
    else if (!e.ollama) pintarEstado("error","error");
    else pintarEstado("listo", `Listo`);
  }

  //mensajes
  const cercaDelFinal = () => log.scrollHeight - log.scrollTop - log.clientHeight < 120;
  const bajar = (forzar) => { if (forzar || cercaDelFinal()) log.scrollTop = log.scrollHeight; };

  function quitarBienvenida() { log.querySelector(".bienvenida")?.remove(); }

  function mensajeUsuario(texto) {
    quitarBienvenida();
    const li = el("li", "mensaje mensaje--usuario");
    li.append(el("p", "burbuja", texto));
    log.append(li);
    bajar(true);
  }

  function mensajeAsistente() {
    const li = el("li", "mensaje mensaje--panchito escribiendo");
    const col = el("div", "mensaje__col");
    const cuerpo = el("div", "burbuja");
    const piezas = el("div", "piezas");
    piezas.style.display = "contents";
    const pensando = el("div", "pensando");
    const puntos = el("span", "puntos");
    puntos.append(el("i"), el("i"), el("i"));
    pensando.append(puntos, el("span", "", "Panchito está pensando…"));
    const texto = el("p", "texto");
    cuerpo.append(piezas, pensando, texto);
    const meta = el("p", "mensaje__meta");
    const agente = el("b", "", "Panchito");
    const reloj = el("span", "", " · 0 s");
    meta.append(agente, reloj);
    col.append(cuerpo, meta);
    li.append(icono("panchito", "panchito panchito--avatar"), col);
    log.append(li);
    bajar(true);
    const t0 = performance.now();
    const intervalo = setInterval(() => { reloj.textContent = ` · ${Math.round((performance.now() - t0) / 1000)} s`; }, 1000);
    return { li, agente, reloj, cuerpo, piezas, pensando, texto, intervalo, finales: el("div") };
  }

  //piezas de la respuesta
  function nivelCarga(cg) { return cg <= 10 ? "baja" : cg < 20 ? "media" : "alta"; }

  function tablaNutrimental(datos) {
    const fig = $("#t-tabla").content.firstElementChild.cloneNode(true);
    const tbody = fig.querySelector("tbody");
    for (const f of datos.filas) {
      const tr = el("tr");
      const th = el("th", "", f.nombre.charAt(0).toUpperCase() + f.nombre.slice(1));
      th.scope = "row";
      const cantidad = f.cantidad.endsWith(" g") || f.cantidad.endsWith(" ml") ? f.cantidad : `${f.cantidad} · ${num(f.gramos, 0)} g`;
      th.append(el("small", "", f.estimado ? `${cantidad} (porción estándar)` : cantidad));
      tr.append(th, el("td", "kcal", num(f.kcal, 0)), el("td", "", `${num(f.carbohidratos)} g`),
        el("td", "", `${num(f.proteina)} g`), el("td", "", `${num(f.grasa)} g`), el("td", "", `${num(f.fibra)} g`));
      tbody.append(tr);
    }
    if (datos.filas.length > 1) {
      const t = datos.total;
      const tr = el("tr");
      const th = el("th", "", "Total");
      th.scope = "row";
      tr.append(th, el("td", "kcal", num(t.kcal, 0)), el("td", "", `${num(t.carbohidratos)} g`),
        el("td", "", `${num(t.proteina)} g`), el("td", "", `${num(t.grasa)} g`), el("td", "", `${num(t.fibra)} g`));
      fig.querySelector("tfoot").append(tr);
    }
    const carga = fig.querySelector(".tabla-nutri__carga");
    const cg = datos.carga_glucemica;
    if (cg !== null && cg !== undefined) {
      if (datos.sin_ig && datos.sin_ig.length) {
        carga.dataset.nivel = "parcial";
        carga.append("Carga glucémica parcial: ", el("strong", "", num(cg)),
          ` (sin dato para ${datos.sin_ig.join(", ")}; la carga real es mayor).`);
      } else {
        const nivel = nivelCarga(cg);
        carga.dataset.nivel = nivel;
        carga.append("Carga glucémica estimada: ", el("strong", "", `${num(cg)} · ${nivel}`));
      }
    }
    return fig;
  }

  function alerta(texto) {
    const caja = el("div", "alerta");
    caja.setAttribute("role", "alert");
    const limpio = texto.replace(/^ALERTA:\s*/i, "");
    caja.append(icono("i-alerta"), el("p", "", limpio.charAt(0).toUpperCase() + limpio.slice(1)));
    const llamar = el("a", "", "911");
    llamar.href = "tel:911";
    llamar.prepend(icono("i-telefono"));
    caja.append(llamar);
    return caja;
  }

  function nota(texto, clase = "") {
    const n = el("p", `nota ${clase}`.trim());
    n.append(icono(clase === "nota--error" ? "i-alerta" : "i-aprox"), el("span", "", texto));
    return n;
  }

  //registro de glucosa
  function estadoGlucosa(v) {
    if (v < 70) return ["bajo", "Baja"];
    if (v > 300) return ["muyalto", "Muy alta"];
    if (v > 180) return ["alto", "Alta"];
    return ["ok", "En rango"];
  }

  function pintarPerfil(perfil = {}) {
    const p = $("#perfil");
    const partes = [];
    if (perfil.tipo) partes.push(["Diabetes", perfil.tipo === "gestacional" ? "gestacional" : `tipo ${perfil.tipo}`]);
    if (perfil.peso_kg) partes.push(["Peso", `${num(perfil.peso_kg)} kg`]);
    if (perfil.estatura_m) partes.push(["Estatura", `${num(perfil.estatura_m, 2)} m`]);
    if (perfil.peso_kg && perfil.estatura_m) partes.push(["IMC", num(perfil.peso_kg / perfil.estatura_m ** 2)]);
    p.replaceChildren();
    if (!partes.length) { p.textContent = "Todavía no me has dicho tu tipo de diabetes, peso ni estatura."; return; }
    partes.forEach(([k, v], i) => { if (i) p.append(" · "); p.append(`${k}: `, el("strong", "", v)); });
  }

  function pintarGlucosas(glucosas = []) {
    const tbody = $("#lecturas tbody");
    tbody.replaceChildren();
    if (!glucosas.length) {
      const tr = el("tr", "vacia");
      const td = el("td", "", "Cuando me digas una glucosa (por ejemplo “salió en 180”), la anoto aquí.");
      td.colSpan = 3;
      tr.append(td);
      tbody.append(tr);
    }
    for (const g of glucosas) {
      const [clave, nombre] = estadoGlucosa(g.mg_dl);
      const tr = el("tr");
      tr.dataset.estado = clave;
      tr.append(el("td", "", (g.fecha || "").slice(11, 16)), el("td", "", `${num(g.mg_dl, 0)} mg/dl`), el("td", "", nombre));
      tbody.append(tr);
    }
    dibujarGrafica(glucosas);
  }

  function dibujarGrafica(glucosas) {
    const caja = $("#grafica");
    const ns = "http://www.w3.org/2000/svg";
    const w = caja.clientWidth || 300;
    const h = caja.clientHeight || 144;
    const izq = 36;
    const y = (v) => h - ((Math.min(Math.max(v, 40), 400) - 40) / 360) * h;
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    const mk = (tag, attrs, texto) => {
      const n = document.createElementNS(ns, tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
      if (texto) n.textContent = texto;
      svg.append(n);
      return n;
    };
    mk("rect", { class: "rango", x: izq, y: y(180), width: w - izq, height: y(70) - y(180) });
    mk("line", { class: "rango-borde", x1: izq, x2: w, y1: y(180), y2: y(180) });
    mk("line", { class: "rango-borde", x1: izq, x2: w, y1: y(70), y2: y(70) });
    mk("line", { class: "umbral", x1: izq, x2: w, y1: y(300), y2: y(300) });
    mk("text", { x: 0, y: y(300) + 4, class: "umbral-texto" }, "300");
    mk("text", { x: izq + 6, y: y(300) - 6, class: "umbral-texto" }, "Alto riesgo");
    mk("text", { x: 0, y: y(180) + 4 }, "180");
    mk("text", { x: 0, y: y(70) + 4 }, "70");
    if (glucosas.length) {
      const paso = (w - izq - 24) / Math.max(glucosas.length - 1, 1);
      const pts = glucosas.map((g, i) => [izq + 12 + (glucosas.length > 1 ? i * paso : (w - izq - 24) / 2), y(g.mg_dl)]);
      if (pts.length > 1) mk("polyline", { class: "linea", points: pts.map((p) => p.join(",")).join(" ") });
      pts.forEach(([px, py], i) => {
        const [clave] = estadoGlucosa(glucosas[i].mg_dl);
        const nueva = i === pts.length - 1 && glucosas.length > glucosasPrevias;
        mk("rect", { class: `marca marca--${clave}${nueva ? " nueva" : ""}`, x: px - 7, y: py - 7, width: 14, height: 14 });
      });
    } else {
      mk("text", { x: izq + 12, y: h / 2 + 4 }, "Rango de referencia general: 70 a 180 mg/dl");
    }
    glucosasPrevias = glucosas.length;
    caja.replaceChildren(svg);
  }

  let ultimas = [];
  window.addEventListener("resize", () => { glucosasPrevias = ultimas.length; dibujarGrafica(ultimas); });

  //envío
  function bloquear(si) {
    ocupado = si;
    caja.disabled = si;
    botonEnviar.disabled = si;
    botonNueva.disabled = si;
    ejemplos.forEach((b) => { b.disabled = si; });
    if (si) pintarEstado("ocupado", "Respondiendo…");
  }

  async function enviar(texto) {
    texto = texto.trim();
    if (!texto || ocupado) return;
    bloquear(true);
    mensajeUsuario(texto);
    caja.value = "";
    caja.style.height = "auto";
    const m = mensajeAsistente();
    const quitarPensando = () => { if (m.pensando.isConnected) m.pensando.remove(); };

    // try/catch: si algo falla al dibujar la respuesta, igual se detiene el reloj y se desbloquea la página
    try {
    await Asistente.enviar(texto, {
      alFijo(p) {
        quitarPensando();
        if (p.clase === "tabla" && p.datos) m.piezas.append(tablaNutrimental(p.datos));
        else if (p.clase === "emergencia") m.piezas.append(alerta(p.texto));
        else if (p.clase === "fuera") m.piezas.append(nota(p.texto, "nota--fuera"));
        else m.finales.append(nota(p.texto));   // estimado: va al final de la respuesta
        bajar();
      },
      alTexto(parte) {
        quitarPensando();
        m.texto.textContent += parte;
        bajar();
      },
      alFin(fin) {
        // limpieza final
        m.texto.textContent = m.texto.textContent.replace(/^[ \t]*\*[ \t]+/gm, "- ").replace(/\n{3,}/g, "\n\n").trim();
        m.agente.textContent = fin.agente && fin.agente !== "-" ? `Panchito · ${fin.agente}` : "Panchito";
        m.reloj.textContent = ` · ${num(fin.segundos)} s`;
        const tokens = fin.tokens || { respuesta: 0, entrada: 0, salida: 0 };
        m.reloj.textContent += ` · ${num(tokens.respuesta, 0)} tokens`;
        m.reloj.title = `Entrada: ${num(tokens.entrada, 0)} · Salida: ${num(tokens.salida, 0)} (incluye la clasificación)`;
        ultimas = fin.glucosas || [];
        pintarPerfil(fin.perfil);
        pintarGlucosas(ultimas);
      },
      alError(msg) {
        quitarPensando();
        m.agente.textContent = "Panchito · error";
        m.cuerpo.append(nota(msg, "nota--error"));
      },
    });
    } catch (e) {
      console.error(e);
      m.cuerpo.append(nota(`Error al mostrar la respuesta: ${e.message}`, "nota--error"));
    }

    clearInterval(m.intervalo);
    quitarPensando();
    m.li.classList.remove("escribiendo");
    if (!m.texto.textContent.trim()) m.texto.remove();
    m.cuerpo.append(...m.finales.childNodes);
    bajar();
    bloquear(false);
    revisarEstado();
    caja.focus();
  }

  const ajustarAltura = () => { caja.style.height = "auto"; caja.style.height = `${caja.scrollHeight}px`; };
  caja.addEventListener("input", ajustarAltura);

  form.addEventListener("submit", (e) => { e.preventDefault(); enviar(caja.value); });
  caja.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); enviar(caja.value); }
  });
  ejemplos.forEach((b) => b.addEventListener("click", () => enviar(b.textContent)));

  botonNueva.addEventListener("click", async () => {
    if (ocupado) return;
    try {
      await Asistente.nuevaConversacion();
    } catch {
      log.append(nota("error de servidor", "error"));
      return;
    }
    log.innerHTML = bienvenida;
    ultimas = [];
    glucosasPrevias = 0;
    pintarPerfil();
    pintarGlucosas();
    caja.focus();
  });

  pintarTips();
  pintarGlucosas();
  Asistente.nuevaConversacion().catch(() => {});   // recargar
  revisarEstado();
  setInterval(revisarEstado, 15000);
})();
