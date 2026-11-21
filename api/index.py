from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = FastAPI(title="Neuro-Fuzzy Architecture DSS (Labs 1 & 2)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# МОДУЛЬ ДЛЯ ЛАБОРАТОРНОЇ РОБОТИ № 2: РОБОЧА ПАМ'ЯТЬ ФАКТІВ (CLIPS-STYLE)
# =====================================================================
TEMPLATES_SCHEMA = {
    "neural_architecture": ["name", "family", "flops_score", "latency_score", "memory_score", "multimodal"],
    "tz_requirement": ["domain", "max_latency", "min_robustness"],
    "hardware_constraint": ["platform", "ram_mb", "accelerator"]
}

INITIAL_DEFFACTS = [
    {"template": "neural_architecture", "slots": {"name": "MobileNetV3-Small", "family": "Depthwise_CNN", "flops_score": 20, "latency_score": 15, "memory_score": 25, "multimodal": "no"}},
    {"template": "neural_architecture", "slots": {"name": "ResNet-18", "family": "Residual_CNN", "flops_score": 55, "latency_score": 45, "memory_score": 50, "multimodal": "no"}},
    {"template": "neural_architecture", "slots": {"name": "all-MiniLM-L6-v2", "family": "Distilled_Transformer", "flops_score": 40, "latency_score": 30, "memory_score": 35, "multimodal": "no"}},
    {"template": "neural_architecture", "slots": {"name": "CLIP-ViT-B/32", "family": "Multimodal_Fusion", "flops_score": 80, "latency_score": 70, "memory_score": 85, "multimodal": "yes"}},
    {"template": "tz_requirement", "slots": {"domain": "Computer_Vision_Edge", "max_latency": 25, "min_robustness": 80}},
    {"template": "hardware_constraint", "slots": {"platform": "Raspberry_Pi_4", "ram_mb": 4096, "accelerator": "CPU_ARM"}}
]

WORKING_MEMORY: List[Dict[str, Any]] = []
NEXT_FACT_ID = 1


def reset_facts_memory():
    global WORKING_MEMORY, NEXT_FACT_ID
    WORKING_MEMORY = []
    NEXT_FACT_ID = 1
    for item in INITIAL_DEFFACTS:
        WORKING_MEMORY.append({
            "id": f"f-{NEXT_FACT_ID}",
            "num_id": NEXT_FACT_ID,
            "template": item["template"],
            "slots": dict(item["slots"])
        })
        NEXT_FACT_ID += 1


reset_facts_memory()


class AssertFactRequest(BaseModel):
    template: str
    slots: Dict[str, Any]


class ModifyFactRequest(BaseModel):
    num_id: int
    slot_name: str
    new_value: Any


class RetractFactRequest(BaseModel):
    num_ids: List[int]


@app.get("/api/facts")
def get_facts():
    return {
        "templates": TEMPLATES_SCHEMA,
        "count": len(WORKING_MEMORY),
        "facts": WORKING_MEMORY
    }


@app.post("/api/facts/reset")
def api_reset_facts():
    reset_facts_memory()
    return {"message": "Виконано (reset): завантажено початкові факти з deffacts (f-1..f-6).", "facts": WORKING_MEMORY}


@app.post("/api/facts/assert")
def api_assert_fact(req: AssertFactRequest):
    global NEXT_FACT_ID
    if req.template not in TEMPLATES_SCHEMA:
        raise HTTPException(status_code=400, detail=f"Невідомий шаблон {req.template}")
    new_fact = {
        "id": f"f-{NEXT_FACT_ID}",
        "num_id": NEXT_FACT_ID,
        "template": req.template,
        "slots": req.slots
    }
    WORKING_MEMORY.append(new_fact)
    NEXT_FACT_ID += 1
    return {"message": f"Виконано (assert): додано факт <Fact-{new_fact['num_id']}>.", "fact": new_fact, "facts": WORKING_MEMORY}


@app.post("/api/facts/modify")
def api_modify_fact(req: ModifyFactRequest):
    for f in WORKING_MEMORY:
        if f["num_id"] == req.num_id:
            if req.slot_name not in f["slots"]:
                raise HTTPException(status_code=400, detail=f"Слот {req.slot_name} відсутній у шаблоні {f['template']}")
            f["slots"][req.slot_name] = req.new_value
            return {"message": f"Виконано (modify {req.num_id} ({req.slot_name} {req.new_value})): <Fact-{req.num_id}> оновлено.", "facts": WORKING_MEMORY}
    raise HTTPException(status_code=404, detail=f"Факт f-{req.num_id} не знайдено")


@app.post("/api/facts/retract")
def api_retract_facts(req: RetractFactRequest):
    global WORKING_MEMORY
    WORKING_MEMORY = [f for f in WORKING_MEMORY if f["num_id"] not in req.num_ids]
    return {"message": f"Виконано (retract {' '.join(map(str, req.num_ids))}): видалено вказані індекси.", "facts": WORKING_MEMORY}


@app.post("/api/facts/clear")
def api_clear_facts():
    global WORKING_MEMORY, NEXT_FACT_ID
    WORKING_MEMORY = []
    NEXT_FACT_ID = 1
    return {"message": "Виконано (clear): робочу пам'ять повністю очищено.", "facts": WORKING_MEMORY}


# =====================================================================
# МОДУЛЬ ДЛЯ ЛАБОРАТОРНОЇ РОБОТИ № 1: БАЗА ЗНАНЬ ТА НЕЧІТКЕ ВИВЕДЕННЯ
# =====================================================================
def synthesize_architecture_from_tz(tz_text: str, domain: str, x1: float, x2: float, x5: float, x10: float):
    text = f"{tz_text} {domain}".lower()

    if x10 >= 70 or any(k in text for k in ["мультимодал", "multimodal", "текст і фото", "аудіо та відео"]):
        arch_type = "Multimodal Cross-Attention Fusion Network"
        auto_benchmark = "CLIP-ViT-B/32"
        layers = [
            {"name": "Input_Visual", "type": "Image Tensor", "shape": "(B, 3, 224, 224)", "params": "0"},
            {"name": "Input_Text", "type": "Token IDs", "shape": "(B, 128)", "params": "0"},
            {"name": "Vision_Encoder", "type": "MobileViT Block x3", "shape": "(B, 256)", "params": "1.8M"},
            {"name": "Text_Encoder", "type": "TinyBERT Layers x4", "shape": "(B, 256)", "params": "4.2M"},
            {"name": "Cross_Attention", "type": "Multi-Head Fusion (4 heads)", "shape": "(B, 512)", "params": "1.1M"},
            {"name": "Classifier_Head", "type": "Linear + Softmax", "shape": "(B, num_classes)", "params": "0.1M"}
        ]
        rationale = "Виявлено мультимодальні вимоги (x10 >= 70 або ключові слова). Обрано гібридну архітектуру з механізмом перехресної уваги."

    elif any(k in text for k in ["текст", "nlp", "чат", "переклад", "мова", "токени", "bert", "gpt"]):
        arch_type = "Compact Transformer Encoder (Distilled Self-Attention)"
        auto_benchmark = "all-MiniLM-L6-v2"
        layers = [
            {"name": "Input_Tokens", "type": "Embedding + PosEnc", "shape": "(B, 256, 384)", "params": "11.7M"},
            {"name": "Transformer_Block_1", "type": "Multi-Head Attention + FFN", "shape": "(B, 256, 384)", "params": "1.8M"},
            {"name": "Transformer_Block_2", "type": "Multi-Head Attention + FFN", "shape": "(B, 256, 384)", "params": "1.8M"},
            {"name": "Pooling_Layer", "type": "Mean Pooling + LayerNorm", "shape": "(B, 384)", "params": "0.01M"},
            {"name": "Output_Head", "type": "Dense Projection", "shape": "(B, out_dim)", "params": "0.05M"}
        ]
        rationale = "Задача належить до обробки природної мови (NLP). Оптимальним є компактний енкодер Трансформера з дистильованими шарами уваги."

    elif any(k in text for k in ["часові ряди", "прогноз", "сенсор", "сигнал", "аудіо", "послідовн", "time series"]):
        arch_type = "1D-CNN + BiGRU Hybrid Recurrent Network"
        auto_benchmark = "TCN-Baseline"
        layers = [
            {"name": "Input_Sequence", "type": "1D Signal Tensor", "shape": "(B, channels, seq_len)", "params": "0"},
            {"name": "Conv1D_Extractor", "type": "Conv1D (k=5, s=1) + BatchNorm + GELU", "shape": "(B, 64, seq_len)", "params": "12K"},
            {"name": "BiGRU_Core", "type": "Bidirectional GRU (hidden=128)", "shape": "(B, seq_len, 256)", "params": "148K"},
            {"name": "Temporal_Attention", "type": "Additive Attention", "shape": "(B, 256)", "params": "32K"},
            {"name": "Regression_Head", "type": "Linear Output", "shape": "(B, horizon)", "params": "2K"}
        ]
        rationale = "Для послідовних даних та сигналів гібрид 1D-згорток та двонаправлених рекурентних блоків (BiGRU) забезпечує мінімальну затримку."

    else:
        if x1 <= 40 and x2 <= 40:
            arch_type = "Depthwise Separable CNN (Inverted Residual / Edge-Optimized)"
            auto_benchmark = "MobileNetV3-Small"
            layers = [
                {"name": "Input_Image", "type": "RGB Tensor", "shape": "(B, 3, 224, 224)", "params": "0"},
                {"name": "Stem_Conv", "type": "Conv2D 3x3 (stride=2) + HardSwish", "shape": "(B, 16, 112, 112)", "params": "464"},
                {"name": "IRB_Block_1", "type": "Depthwise Conv 3x3 + SE-Attention", "shape": "(B, 24, 56, 56)", "params": "3.8K"},
                {"name": "IRB_Block_2", "type": "Depthwise Conv 5x5 + SE-Attention", "shape": "(B, 48, 28, 28)", "params": "18.4K"},
                {"name": "IRB_Block_3", "type": "Inverted Residual (exp=6)", "shape": "(B, 96, 14, 14)", "params": "64.2K"},
                {"name": "Global_Pool", "type": "AdaptiveAvgPool2d + Dropout(0.2)", "shape": "(B, 96, 1, 1)", "params": "0"},
                {"name": "Classifier", "type": "Linear (96 -> num_classes)", "shape": "(B, num_classes)", "params": "9.7K"}
            ]
            rationale = "Жорсткі вимоги до обчислювальної складності (x1 <= 40) та затримки (x2 <= 40). Застосовано інвертовані залишкові блоки з роздільною згорткою (Depthwise Separable)."
        else:
            arch_type = "Residual Convolutional Network with Squeeze-and-Excitation (SE-ResNet)"
            auto_benchmark = "ResNet-18"
            layers = [
                {"name": "Input_Tensor", "type": "Feature Map", "shape": "(B, 3, 224, 224)", "params": "0"},
                {"name": "Conv1_Stem", "type": "Conv2D 7x7 + BatchNorm + ReLU", "shape": "(B, 64, 112, 112)", "params": "9.4K"},
                {"name": "ResStage_1", "type": "2x Residual Block + SE-Gate", "shape": "(B, 64, 56, 56)", "params": "148K"},
                {"name": "ResStage_2", "type": "2x Residual Block (stride=2)", "shape": "(B, 128, 28, 28)", "params": "526K"},
                {"name": "ResStage_3", "type": "2x Residual Block (stride=2)", "shape": "(B, 256, 14, 14)", "params": "2.1M"},
                {"name": "Head", "type": "GlobalAvgPool + Linear", "shape": "(B, num_classes)", "params": "256K"}
            ]
            rationale = "Дозволено підвищені ресурсні витрати для досягнення високої робастності (x5). Застосовано залишкові зв'язки (Skip-Connections) та канали уваги SE."

    return {
        "recommended_architecture": arch_type,
        "suggested_benchmark": auto_benchmark,
        "rationale": rationale,
        "blueprint_layers": layers
    }


def build_fuzzy_engine():
    terms = ['Н', 'нС', 'С', 'вС', 'В']

    x1_complexity = ctrl.Antecedent(np.arange(0, 101, 1), 'x1_complexity')
    x2_latency = ctrl.Antecedent(np.arange(0, 101, 1), 'x2_latency')
    x3_memory = ctrl.Antecedent(np.arange(0, 101, 1), 'x3_memory')
    x5_robustness = ctrl.Antecedent(np.arange(0, 101, 1), 'x5_robustness')
    x7_generalization = ctrl.Antecedent(np.arange(0, 101, 1), 'x7_generalization')
    x10_multimodal = ctrl.Antecedent(np.arange(0, 101, 1), 'x10_multimodal')

    for var in [x1_complexity, x2_latency, x3_memory, x5_robustness, x7_generalization, x10_multimodal]:
        var.automf(names=terms)

    sing = ctrl.Consequent(np.arange(-1.0, 1.05, 0.05), 'sing')
    sing['Критично_гірше'] = fuzz.trimf(sing.universe, [-1.0, -1.0, -0.6])
    sing['Гірше'] = fuzz.trimf(sing.universe, [-0.8, -0.45, -0.2])
    sing['Еквівалентно'] = fuzz.trimf(sing.universe, [-0.3, 0.0, 0.3])
    sing['Краще'] = fuzz.trimf(sing.universe, [0.2, 0.5, 0.8])
    sing['Ідеально'] = fuzz.trimf(sing.universe, [0.6, 1.0, 1.0])

    rules = [
        ctrl.Rule(
            x1_complexity['Н'] & x2_latency['Н'] & x3_memory['Н'] &
            x5_robustness['В'] & x7_generalization['В'],
            sing['Ідеально']
        ),
        ctrl.Rule(
            (x1_complexity['нС'] | x1_complexity['С']) &
            x2_latency['нС'] & x5_robustness['вС'] & x7_generalization['вС'],
            sing['Краще']
        ),
        ctrl.Rule(
            x1_complexity['С'] & x2_latency['С'] & x5_robustness['С'] & x7_generalization['С'],
            sing['Еквівалентно']
        ),
        ctrl.Rule(
            x1_complexity['В'] | x2_latency['В'] | x3_memory['В'],
            sing['Гірше']
        ),
        ctrl.Rule(
            x5_robustness['Н'] | x7_generalization['Н'],
            sing['Критично_гірше']
        ),
        ctrl.Rule(
            x10_multimodal['В'] & x5_robustness['вС'] & x2_latency['нС'],
            sing['Краще']
        )
    ]

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


class EvaluationRequest(BaseModel):
    tz_text: str = Field(..., example="Розпізнавання облич на Raspberry Pi в реальному часі...")
    domain: str = Field(..., example="Computer Vision (Edge / Raspberry Pi)")
    benchmark_name: str = Field(..., example="MobileNetV3-Small")
    x1_complexity: float = Field(..., ge=0, le=100)
    x2_latency: float = Field(..., ge=0, le=100)
    x3_memory: float = Field(..., ge=0, le=100)
    x5_robustness: float = Field(..., ge=0, le=100)
    x7_generalization: float = Field(..., ge=0, le=100)
    x10_multimodal: float = Field(..., ge=0, le=100)


@app.post("/api/evaluate")
def evaluate_model(req: EvaluationRequest):
    sim = build_fuzzy_engine()

    sim.input['x1_complexity'] = req.x1_complexity
    sim.input['x2_latency'] = req.x2_latency
    sim.input['x3_memory'] = req.x3_memory
    sim.input['x5_robustness'] = req.x5_robustness
    sim.input['x7_generalization'] = req.x7_generalization
    sim.input['x10_multimodal'] = req.x10_multimodal

    try:
        sim.compute()
        sing_score = float(sim.output['sing'])
        fallback_used = False
    except Exception:
        sing_score = 0.0
        fallback_used = True

    if sing_score >= 0.6:
        verdict = "Ідеально / SOTA"
        status_code = "ideal"
    elif sing_score >= 0.2:
        verdict = "Краще за еталон (Рекомендовано)"
        status_code = "better"
    elif sing_score >= -0.2:
        verdict = "Еквівалентно еталону (Норма)"
        status_code = "neutral"
    elif sing_score >= -0.6:
        verdict = "Гірше (Потребує оптимізації)"
        status_code = "worse"
    else:
        verdict = "Критично гірше (Відхилити)"
        status_code = "critical"

    arch_synthesis = synthesize_architecture_from_tz(
        req.tz_text, req.domain, req.x1_complexity, req.x2_latency, req.x5_robustness, req.x10_multimodal
    )

    active_arch_facts = [f["slots"]["name"] for f in WORKING_MEMORY if f["template"] == "neural_architecture"]

    trace = [
        f"Аналіз ТЗ: '{req.tz_text[:75]}...' -> Рекомендовано сімейство: {arch_synthesis['recommended_architecture']}.",
        arch_synthesis['rationale'],
        f"Активні еталони в робочій пам'яті ({len(active_arch_facts)} шт.): {', '.join(active_arch_facts) if active_arch_facts else 'немає (пам’ять очищена)'}."
    ]
    if req.x1_complexity > 70 or req.x2_latency > 70:
        trace.append(f"Високі ресурсні витрати: складність x1={req.x1_complexity}, затримка x2={req.x2_latency} знижують груповий індекс ефективності (y).")
    else:
        trace.append(f"Ресурсна ефективність (y) у межах норми (x1={req.x1_complexity}, x2={req.x2_latency}, x3={req.x3_memory}).")

    if req.x5_robustness < 35 or req.x7_generalization < 35:
        trace.append(f"Критично низька надійність (z): стійкість x5={req.x5_robustness}, узагальнення x7={req.x7_generalization}.")
    else:
        trace.append(f"Надійність моделі (z) підтверджена (стійкість x5={req.x5_robustness}, узагальнення x7={req.x7_generalization}).")

    generative_prob = 0.65 if fallback_used else 0.15
    warning_msg = (
        "УВАГА: Спрацював евристичний компенсатор (дозволена ймовірність генерації > 50%). Необхідне оновлення бази правил через науковий краулер."
        if fallback_used else
        "Доказова база: детерміновані нечіткі правила (Лаба №1). Частка генеративної евристики в межах норми (15%)."
    )

    return {
        "sing_score": round(sing_score, 3),
        "verdict": verdict,
        "status_code": status_code,
        "benchmark": req.benchmark_name,
        "architecture_synthesis": arch_synthesis,
        "reasoning_trace": trace,
        "knowledge_audit": {
            "generative_probability": generative_prob,
            "allowed_threshold": 0.50,
            "warning": warning_msg
        }
    }


# Підключення фронтенду для локального запуску
public_dir = os.path.join(os.path.dirname(__file__), "..", "public")
if os.path.isdir(public_dir):
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="public")