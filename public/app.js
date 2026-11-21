// Перемикання вкладок: приховує неактивну частину та відкриває активну вгорі
function focusPart(partNum) {
    const btn1 = document.getElementById('btn-part1');
    const btn2 = document.getElementById('btn-part2');
    const part1 = document.getElementById('part-1-section');
    const part2 = document.getElementById('part-2-section');

    if (partNum === 1) {
        btn1.classList.add('active');
        btn2.classList.remove('active');
        part1.style.display = 'block';
        part2.style.display = 'none';
    } else {
        btn2.classList.add('active');
        btn1.classList.remove('active');
        part1.style.display = 'none';
        part2.style.display = 'block';
        renderSlotInputs();
        loadFacts();
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function updateVal(id) {
    document.getElementById(`val-${id}`).textContent = document.getElementById(id).value;
}

// ==================== ЛОГІКА ЧАСТИНИ 1 ====================
document.getElementById('evaluate-btn').addEventListener('click', async () => {
    const btn = document.getElementById('evaluate-btn');
    btn.disabled = true;
    btn.textContent = "Обчислення нечіткого виведення...";

    const payload = {
        tz_text: document.getElementById('tz_text').value,
        domain: document.getElementById('domain').value,
        benchmark_name: document.getElementById('benchmark_name').value,
        x1_complexity: parseFloat(document.getElementById('x1').value),
        x2_latency: parseFloat(document.getElementById('x2').value),
        x3_memory: parseFloat(document.getElementById('x3').value),
        x5_robustness: parseFloat(document.getElementById('x5').value),
        x7_generalization: parseFloat(document.getElementById('x7').value),
        x10_multimodal: parseFloat(document.getElementById('x10').value),
    };

    try {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("Помилка сервера при розрахунку.");
        const data = await response.json();
        updateDashboard(data);
    } catch (error) {
        alert("Не вдалося виконати розрахунок: " + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Спроєктувати архітектуру та оцінити";
    }
});

function updateDashboard(data) {
    const score = data.sing_score;
    const scoreEl = document.getElementById('sing-value');
    scoreEl.textContent = (score > 0 ? "+" : "") + score.toFixed(3);

    const percent = ((score + 1) / 2) * 100;
    document.getElementById('scale-pointer').style.left = `${Math.max(2, Math.min(98, percent))}%`;

    const banner = document.getElementById('verdict-banner');
    banner.textContent = data.verdict;
    banner.className = `verdict-banner ${data.status_code}`;

    const audit = data.knowledge_audit;
    const probPercent = Math.round(audit.generative_probability * 100);
    const genProbEl = document.getElementById('gen-prob');
    genProbEl.textContent = `${probPercent}%`;
    genProbEl.style.color = audit.generative_probability > audit.allowed_threshold ? '#ef4444' : '#10b981';
    document.getElementById('audit-warning').textContent = audit.warning.replace('(Лаба №1)', '(Частина 1)');

    const traceList = document.getElementById('reasoning-list');
    traceList.innerHTML = '';
    data.reasoning_trace.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        traceList.appendChild(li);
    });

    const arch = data.architecture_synthesis;
    document.getElementById('arch-title').textContent = arch.recommended_architecture;
    document.getElementById('benchmark-badge').textContent = `Рекомендований еталон: ${arch.suggested_benchmark}`;

    const diagramContainer = document.getElementById('diagram-container');
    diagramContainer.innerHTML = '';

    let specLines = [
        `# АРХІТЕКТУРА: ${arch.recommended_architecture}`,
        `# ЕТАЛОН ПОРІВНЯННЯ: ${arch.suggested_benchmark}`,
        `# ІНДЕКС ЕФЕКТИВНОСТІ (Sing): ${(score > 0 ? "+" : "") + score.toFixed(3)} [${data.verdict}]`,
        `# ------------------------------------------------------------------`,
        `# Шар (Layer)              | Тип блоку                      | Розмірність тензора   | Параметри`,
        `# ------------------------------------------------------------------`
    ];

    arch.blueprint_layers.forEach((layer, idx) => {
        const node = document.createElement('div');
        node.className = 'layer-node';
        node.innerHTML = `
            <div>
                <div class="layer-name">${idx + 1}. ${layer.name}</div>
                <div class="layer-type">${layer.type}</div>
            </div>
            <div class="layer-meta">
                <span class="layer-shape">${layer.shape}</span>
                <span class="layer-params">${layer.params} params</span>
            </div>
        `;
        diagramContainer.appendChild(node);

        if (idx < arch.blueprint_layers.length - 1) {
            const arrow = document.createElement('div');
            arrow.className = 'layer-arrow';
            arrow.textContent = '↓';
            diagramContainer.appendChild(arrow);
        }

        const col1 = `${idx + 1}. ${layer.name}`.padEnd(24);
        const col2 = layer.type.padEnd(30);
        const col3 = layer.shape.padEnd(21);
        specLines.push(`${col1} | ${col2} | ${col3} | ${layer.params}`);
    });

    document.getElementById('blueprint-spec').textContent = specLines.join('\n');
}

// ==================== ЛОГІКА ЧАСТИНИ 2 ====================
const TEMPLATE_DEFAULTS = {
    neural_architecture: {
        name: "EfficientNet-B0",
        family: "Compound_Scaled_CNN",
        flops_score: 35,
        latency_score: 28,
        memory_score: 32,
        multimodal: "no"
    },
    tz_requirement: {
        domain: "Medical_Imaging_2D",
        max_latency: 50,
        min_robustness: 90
    },
    hardware_constraint: {
        platform: "NVIDIA_Jetson_Nano",
        ram_mb: 4096,
        accelerator: "GPU_Maxwell_128"
    }
};

function renderSlotInputs() {
    const tpl = document.getElementById('fact-template').value;
    const container = document.getElementById('dynamic-slots-container');
    const defaults = TEMPLATE_DEFAULTS[tpl];
    container.innerHTML = '';

    Object.entries(defaults).forEach(([slotName, defaultVal]) => {
        const div = document.createElement('div');
        div.className = 'input-group';
        div.style.marginBottom = '4px';
        div.innerHTML = `
            <label>${slotName}:</label>
            <input type="text" class="slot-input" data-slot="${slotName}" value="${defaultVal}">
        `;
        container.appendChild(div);
    });
}

function logClips(msg) {
    const logEl = document.getElementById('clips-log');
    const time = new Date().toLocaleTimeString();
    logEl.innerHTML = `[${time}] ${msg}<br>` + logEl.innerHTML;
}

async function loadFacts() {
    try {
        const res = await fetch('/api/facts');
        const data = await res.json();
        renderFactsList(data.facts);
    } catch (e) {
        logClips("Помилка з'єднання з сервером /api/facts");
    }
}

function prefillModify(numId, firstSlot, firstVal) {
    document.getElementById('mod-id').value = numId;
    document.getElementById('mod-slot').value = firstSlot;
    document.getElementById('mod-val').value = firstVal;
}

function renderFactsList(facts) {
    const listEl = document.getElementById('facts-list');
    document.getElementById('facts-count-badge').textContent = `Активних фактів: ${facts.length}`;
    document.getElementById('stat-arch').textContent = facts.filter(f => f.template === 'neural_architecture').length;
    document.getElementById('stat-tz').textContent = facts.filter(f => f.template === 'tz_requirement').length;
    document.getElementById('stat-hw').textContent = facts.filter(f => f.template === 'hardware_constraint').length;

    listEl.innerHTML = '';

    if (facts.length === 0) {
        listEl.innerHTML = '<p class="placeholder-text" style="grid-column: 1 / -1;">Робоча пам’ять порожня (0 фактів). Натисніть (reset) для завантаження deffacts.</p>';
        return;
    }

    facts.forEach(f => {
        const entries = Object.entries(f.slots);
        const rowsHtml = entries
            .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
            .join('');

        const [firstKey, firstVal] = entries[0] || ['', ''];

        const card = document.createElement('div');
        card.className = `fact-card tpl-${f.template}`;
        card.innerHTML = `
            <div class="fact-card-top">
                <span class="fact-id-tag">${f.id} &lt;Fact-${f.num_id}&gt;</span>
                <span class="fact-tpl-tag">${f.template}</span>
            </div>
            <table class="fact-slots-table">
                <tbody>${rowsHtml}</tbody>
            </table>
            <div class="fact-card-actions">
                <button class="mini-btn" onclick="prefillModify(${f.num_id}, '${firstKey}', '${firstVal}')">modify</button>
                <button class="mini-btn del" onclick="retractSingleFact(${f.num_id})">retract ${f.num_id}</button>
            </div>
        `;
        listEl.appendChild(card);
    });
}

async function assertNewFact() {
    const tpl = document.getElementById('fact-template').value;
    const slots = {};
    document.querySelectorAll('.slot-input').forEach(inp => {
        const raw = inp.value.trim();
        slots[inp.dataset.slot] = isNaN(raw) || raw === '' ? raw : Number(raw);
    });

    const res = await fetch('/api/facts/assert', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ template: tpl, slots })
    });
    const data = await res.json();
    logClips(data.message);
    renderFactsList(data.facts);
}

async function modifyExistingFact() {
    const num_id = parseInt(document.getElementById('mod-id').value);
    const slot_name = document.getElementById('mod-slot').value.trim();
    const rawVal = document.getElementById('mod-val').value.trim();
    const new_value = isNaN(rawVal) || rawVal === '' ? rawVal : Number(rawVal);

    const res = await fetch('/api/facts/modify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_id, slot_name, new_value })
    });
    const data = await res.json();
    if (!res.ok) {
        logClips(`ПОМИЛКА: ${data.detail}`);
        return;
    }
    logClips(data.message);
    renderFactsList(data.facts);
}

async function retractSingleFact(numId) {
    const res = await fetch('/api/facts/retract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_ids: [numId] })
    });
    const data = await res.json();
    logClips(data.message);
    renderFactsList(data.facts);
}

async function retractOddFacts() {
    const resFacts = await fetch('/api/facts');
    const current = await resFacts.json();
    const oddIds = current.facts.filter(f => f.num_id % 2 !== 0).map(f => f.num_id);
    if (oddIds.length === 0) return;

    const res = await fetch('/api/facts/retract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_ids: oddIds })
    });
    const data = await res.json();
    logClips(data.message);
    renderFactsList(data.facts);
}

async function resetFacts() {
    const res = await fetch('/api/facts/reset', { method: 'POST' });
    const data = await res.json();
    logClips(data.message);
    renderFactsList(data.facts);
}

async function clearFacts() {
    const res = await fetch('/api/facts/clear', { method: 'POST' });
    const data = await res.json();
    logClips(data.message);
    renderFactsList(data.facts);
}

renderSlotInputs();
loadFacts();