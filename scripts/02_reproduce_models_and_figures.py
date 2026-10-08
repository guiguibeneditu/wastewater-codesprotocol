# -*- coding: utf-8 -*-
# ======================================================================================
# REPRODUÇÃO FINAL — WATER RESEARCH
# ROLLING OOS 2023 CONGELADO + RETREINO FINAL + LATÊNCIA FIXED-MODEL + FIGURAS 7–9
# ======================================================================================
# Execute esta célula/arquivo inteiro em um runtime novo do Google Colab.
#
# O que é REUTILIZADO (não recalculado):
#   - somente o rolling causal OOS de 2023, checkpoint a4dcc3a5c1cb, porque ele já é
#     o conjunto cronológico auditado usado para treinar o residual-learning do XGBoost.
#
# O que é TREINADO NOVAMENTE neste run:
#   - SARIMAX final em 2022–2023;
#   - XGBoost residual + lambda contextual;
#   - ensemble LSTM-Q (5 seeds) em 2022–2023.
#
# Experimento de latência:
#   - após o retreino final, modelos, scalers, hiperparâmetros e lambdas ficam CONGELADOS;
#   - somente a informação hidráulica é envelhecida em d = 0,1,2,3,4,5,6,8,12,24 h;
#   - exógenas hidrometeorológicas disponíveis em t não são atrasadas;
#   - os mesmos 8.784 alvos de 2024 são usados em todos os pontos.
#
# O código aborta se o SARIMAX anual recém-treinado não reproduzir o regime final de
# alto desempenho (NSE aproximadamente 0.9327) ou se d=0 não reproduzir as previsões
# anuais recém-geradas.
# ======================================================================================

# ============================================================
# CÉLULA ÚNICA PARA GOOGLE COLAB — ROLLING CAUSAL T+1 COM CHECKPOINT
# COPIE TUDO E EXECUTE.
#
# OBJETIVO OPERACIONAL:
#   emitir Q_{t+1|t} usando exclusivamente informações disponíveis até t.
#
# PROTOCOLO:
#   - 2022: janela inicial de estimação do SARIMAX;
#   - 2023: resíduos OOS gerados por janela expansiva, com refit a cada 168 h;
#   - dentro de cada bloco de 168 h: previsões estritamente de um passo,
#     assimilando Q_tau somente DEPOIS de emitir a previsão de Q_tau;
#   - 2024: teste cronológico independente, sem ajuste de parâmetros,
#     seleção de variáveis ou calibração de hiperparâmetros;
#   - todas as exógenas observacionais são deslocadas em 1 h: a linha-alvo
#     tau=t+1 recebe X_t.
#
# CHECKPOINT:
#   - salvo atomicamente após cada bloco em Parquet + JSON no Google Drive;
#   - ao reiniciar a célula, blocos completos são conferidos e ignorados;
#   - checkpoints com configuração/dados diferentes nunca são reutilizados;
#   - quando as 8.760 h de 2023 estão completas, o pipeline segue sozinho
#     para SARIMAX final, XGBoost, lambda contextual, 2024 e figuras.
#
# ARQUIVOS ESPERADOS EM Meu Drive/Colab Notebooks/:
#   1) IMPUTAÇÃO A SER UTILIZADA.xlsx
#   2) checkpoints_rolling_causal_T1/oos_2023_rolling_causal_T1_a4dcc3a5c1cb.parquet
#   3) checkpoints_rolling_causal_T1/oos_2023_rolling_causal_T1_a4dcc3a5c1cb.json
#
# NÃO precisa trazer nenhum arquivo/modelo LSTM de outra conta: a LSTM-Q é
# retreinada neste próprio run, a partir da mesma série y usada pelos demais modelos.
# ============================================================

import sys as _sys
import subprocess as _subprocess

print("Instalando dependências do modelo...")
_subprocess.check_call([
    _sys.executable, "-m", "pip", "install", "-q",
    "statsmodels==0.14.4",
    "xgboost==2.1.4",
    "pyarrow==18.1.0",
    "openpyxl==3.1.5",
    "tensorflow==2.20.0",
    "keras==3.13.2",
    "joblib",
])

# ============================================================
# MODELO CAMPEÃO CORRIGIDO: PREVISÃO CAUSAL Q_{t+1|t}
# MATRIZ DE ESTADO x_t + LAMBDA CONTEXTUAL
# TODAS AS VARIÁVEIS POSICIONADAS EM t PARA PREVER Q_{t+1}
# OOS DE 2023 PRONTO + hard cases + janela 14h
# ============================================================

import os
import json
import hashlib
import gc
import time
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb

from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    drive = None
    IN_COLAB = False

# Mantém avisos numéricos/convergência visíveis; silencia apenas avisos de API.
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 0. CONFIGURAÇÕES GERAIS
# ============================================================

DEFAULT_WORKSPACE_ROOT = (
    Path(__file__).resolve().parents[1]
    if "__file__" in globals()
    else Path.cwd()
)
WORKSPACE_ROOT = Path(
    os.environ.get("SARIMAX_WORKSPACE", str(DEFAULT_WORKSPACE_ROOT))
)
LOCAL_DATA_PATH = WORKSPACE_ROOT / "data" / "input" / "imputed_hourly_dataset.xlsx"
DRIVE_DATA_PATH = Path("/content/drive/MyDrive/Colab Notebooks/IMPUTAÇÃO A SER UTILIZADA.xlsx")
CAMINHO_ARQUIVO = os.environ.get(
    "SARIMAX_DATA_PATH",
    str(DRIVE_DATA_PATH if IN_COLAB else LOCAL_DATA_PATH),
)
COL_DATA = "datetime"
TARGET_COL = "Vazão"

DATA_INICIO_ESTUDO = "2022-01-01 00:00:00"
DATA_CORTE_TESTE   = "2024-01-01 00:00:00"
DATA_FIM_ESTUDO    = "2024-12-31 23:00:00"

DATA_INICIO_ROLLING = "2023-01-01 00:00:00"
DATA_FIM_ROLLING    = "2023-12-31 23:00:00"

ROLLING_BLOCK_HOURS = 168
FORCE_REBUILD_ROLLING = False

SARIMAX_ORDER = (1, 0, 0)
SARIMAX_SEASONAL_ORDER = (1, 1, 1, 24)

# abordagem campeã
APLICAR_CORRECAO_SOMENTE_EM_EVENTOS = True
JANELA_RESPOSTA_CHUVA_HORAS = 14
QUANTIL_ERRO_DIFICIL = 0.60
TOP_K_FEATURES = 15

CLIP_PREVISAO_QUANTIL_INF = 0.01
CLIP_PREVISAO_QUANTIL_SUP = 0.99

# ============================================================
# PROTOCOLO OPERACIONAL: ONLINE 1H COM LAGS CAUSAIS
# ============================================================
# O DataFrame permanece indexado pela hora-alvo tau = t+1. Todas as colunas
# explicativas são deslocadas em uma hora, de modo que a linha tau contenha
# exclusivamente os valores da linha t. Os lags de vazão/resíduo adicionados
# depois também terminam em t. Nunca entram Q_tau, X_tau ou e_tau.
HORA_ATUALIZACAO_DIARIA = None
TAMANHO_PACOTE_VAZAO_HORAS = 24
PROTOCOLO_ONLINE_1H_LAGS_24H = True

# ============================================================
# AJUSTES DE CHUVA E LAMBDA CONTEXTUAL
# ============================================================
# Se False, as janelas de precipitação acumulada/média/máxima não incluem
# a precipitação do próprio instante alvo t. Elas usam P_{t-1}, P_{t-2}, ...
# Isso evita que precip_acum_6h seja automaticamente >= Precipitação_t.
INCLUIR_PRECIP_ALVO_NAS_JANELAS = False

# Decaimento exponencial para chuva recente: P_{t-1} + a P_{t-2} + ...
ALPHA_PRECIP_DECAY = 0.80

# Teste específico desta versão:
# True = as colunas históricas chamadas precip_acum_3h, precip_acum_6h,
# precip_acum_12h e precip_acum_24h passam a carregar MÉDIAS, não somas.
# Os nomes antigos são mantidos apenas para não quebrar o restante do pipeline.
SUBSTITUIR_ACUMULADOS_POR_MEDIAS = True

# Também converte precip_acum_20d para média horária de 20 dias, dividindo por 480 h.
# Isso evita que inter_precip20d e inter_seca_saturacao carreguem o acumulado bruto.
SUBSTITUIR_ACUM20D_POR_MEDIA20D = True
HORAS_20D = 20 * 24

# Lambda contextual:
# - deixa de usar lambda_global como fallback dominante;
# - encolhe lambdas por estado para 1.0, que significa "confie no XGB como está";
# - estados com baixa acurácia de sinal podem ter a correção amortecida.
LAMBDA_PRIOR_NEUTRO = 1.00
LAMBDA_MIN_OBS_ESTADO_COMPLETO = 18
LAMBDA_MIN_OBS_ESTADO_MEDIO = 25
LAMBDA_MIN_OBS_ESTADO_ERRO = 35
LAMBDA_MIN_SIGN_ACC = 0.50
LAMBDA_AMORTECER_SE_SINAL_RUIM = True

# Calibrar lambda usando todas as janelas pós-chuva de 2023, não só hard cases.
# Isso deixa a matriz lambda menos esparsa e mais representativa dos cenários.
CALIBRAR_LAMBDA_EM_TODOS_EVENTOS_2023 = True

# Saídas
DEFAULT_OUTPUT_DIR = (
    Path("/content/drive/MyDrive/Colab Notebooks/saida_modelo_rolling_causal_T1")
    if IN_COLAB
    else WORKSPACE_ROOT / "outputs" / "causal_model" / "run"
)
OUTPUT_DIR = Path(os.environ.get("SARIMAX_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
NOME_EXCEL_SAIDA = "resultados_rolling_causal_T1_2024.xlsx"
NOME_FIGURA_SAIDA = "comparativo_SARIMAX_HIBRIDO_LSTM_2024.png"
NOME_FIGURA_PDF = "comparativo_SARIMAX_HIBRIDO_LSTM_2024.pdf"
NOME_AUDITORIA = "auditoria_causal_T1.json"

DEFAULT_CHECKPOINT_DIR = (
    Path("/content/drive/MyDrive/Colab Notebooks/checkpoints_rolling_causal_T1")
    if IN_COLAB
    else WORKSPACE_ROOT / "data" / "rolling"
)
CHECKPOINT_DIR = Path(
    os.environ.get("SARIMAX_CHECKPOINT_DIR", str(DEFAULT_CHECKPOINT_DIR))
)

# ======================================================================================
# ROLLING CAUSAL JÁ CONCLUÍDO — USO ESTRITO, SEM FALLBACK PARA RECÁLCULO
# ======================================================================================
EXISTING_ROLLING_TAG = "a4dcc3a5c1cb"
EXISTING_ROLLING_PARQUET = CHECKPOINT_DIR / f"oos_2023_rolling_causal_T1_{EXISTING_ROLLING_TAG}.parquet"
EXISTING_ROLLING_JSON = CHECKPOINT_DIR / f"oos_2023_rolling_causal_T1_{EXISTING_ROLLING_TAG}.json"

# Compatibilidade legada: nenhum arquivo externo de LSTM é usado neste run final.
DEFAULT_LSTM_RESULT_PATH = None
LSTM_RESULT_PATH = None

# ============================================================
# 1. CONFIGURAÇÕES DA MATRIZ DE ESTADO x_t E LAMBDA
# ============================================================

# mínimos quadrados por estado, com shrinkage
LAMBDA_MIN_OBS_ESTADO = 25
LAMBDA_SHRINK_STRENGTH = 40
LAMBDA_CLIP_MIN = 0.00
LAMBDA_CLIP_MAX = 1.25

# se True, lambda só atua na janela pós-chuva
APLICAR_LAMBDA_SOMENTE_EM_EVENTOS = True

# ============================================================
# ZOOM
# ============================================================

DATA_INICIO_ZOOM = "2024-02-10 00:00:00"
DATA_FIM_ZOOM    = "2024-02-17 23:00:00"

# ============================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================

def nse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    if den == 0:
        return np.nan
    return 1 - (num / den)

def get_metrics(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float, float]:
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nse_val = nse(y_true, y_pred)
    spear, _ = spearmanr(y_true, y_pred)
    return nse_val, rmse, spear

def print_metrics_table(
    title: str,
    y_true: pd.Series,
    pred_sarimax: pd.Series,
    pred_hybrid: pd.Series,
    label_a: str = "Modelo A",
    label_b: str = "Modelo B",
):
    nse_s, rmse_s, spear_s = get_metrics(y_true, pred_sarimax)
    nse_h, rmse_h, spear_h = get_metrics(y_true, pred_hybrid)

    print("\n" + "=" * 100)
    print(title)
    print(f"{'Metric':<15} | {label_a:<32} | {label_b:<32} | {'Diferença':<15}")
    print("-" * 100)
    print(f"{'NSE':<15} | {nse_s:<32.4f} | {nse_h:<32.4f} | {(nse_h - nse_s):<15.4f}")
    print(f"{'RMSE':<15} | {rmse_s:<32.4f} | {rmse_h:<32.4f} | {(rmse_h - rmse_s):<15.4f}")
    print(f"{'Spearman':<15} | {spear_s:<32.4f} | {spear_h:<32.4f} | {(spear_h - spear_s):<15.4f}")
    print("=" * 100)

def align_all_predictors_for_t_plus_1(X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Mantém o alvo indexado em tau=t+1 e desloca a matriz explicativa completa
    em uma hora. Portanto, X_aligned.loc[tau] == X_raw.loc[tau-1]. Isso inclui
    variáveis meteorológicas, temporais, calendáricas e features já derivadas.
    """
    return X_raw.shift(1)


def fit_sarimax(y_train: pd.Series, X_train: pd.DataFrame):
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=SARIMAX_ORDER,
        seasonal_order=SARIMAX_SEASONAL_ORDER,
        enforce_stationarity=True,
        enforce_invertibility=True
    )
    # Não usar low_memory=True aqui. Esse modo descarta predicted_state e
    # predicted_state_cov, que são necessários para o extend() iniciar 2024.
    # O controle de memória é obtido abaixo ao filtrar todo 2024 em uma única
    # passagem, sem os 8.784 objetos crescentes criados pelo antigo append().
    result = model.fit(disp=False, low_memory=False, method="lbfgs", maxiter=200)
    converged = bool(getattr(result, "mle_retvals", {}).get("converged", True))
    if not converged:
        warnings.warn(
            "O ajuste SARIMAX não declarou convergência. O bloco será salvo com "
            "converged=False no JSON para revisão; nenhuma falha será ocultada.",
            RuntimeWarning,
        )
    return result


def forecast_one_step_with_observed_updates(
    fitted_result,
    y_test: pd.Series,
    X_test: pd.DataFrame,
) -> pd.Series:
    """
    Previsões causais Q_{tau|tau-1} em uma única passagem do filtro.

    ``extend`` filtra 2024 sequencialmente: para cada tau, ``fittedvalues`` é
    calculado antes da assimilação de Q_tau e, portanto, usa somente o estado
    atualizado até tau-1. Os parâmetros estimados em 2022-2023 permanecem
    fixos. Isso é matematicamente equivalente ao antigo laço get_forecast +
    append, mas não recria um objeto SARIMAX crescente a cada uma das 8.784 h.
    """
    y_test = pd.Series(y_test, copy=False).astype(np.float64)
    X_test = pd.DataFrame(X_test, copy=False).astype(np.float64)

    if not y_test.index.equals(X_test.index):
        raise ValueError("y_test e X_test não possuem o mesmo índice horário.")

    result_test = fitted_result.extend(endog=y_test, exog=X_test)
    values = np.asarray(result_test.fittedvalues, dtype=np.float64).reshape(-1)

    if len(values) != len(y_test):
        raise RuntimeError(
            f"O filtro retornou {len(values)} previsões para {len(y_test)} alvos."
        )

    predictions = pd.Series(values, index=y_test.index, name="pred_sarimax")
    if predictions.isna().any():
        raise RuntimeError("A previsão SARIMAX causal de 2024 contém NaN.")

    del result_test, values
    gc.collect()
    print(f"SARIMAX causal 2024 concluído em uma passagem: {len(predictions)} horas")
    return predictions


def _json_hash(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _data_fingerprint(y_ref: pd.Series, X_ref: pd.DataFrame) -> str:
    """Assinatura dos dados que realmente determinam o rolling."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(y_ref, index=True).values.tobytes())
    h.update(pd.util.hash_pandas_object(X_ref, index=True).values.tobytes())
    h.update("|".join(map(str, X_ref.columns)).encode("utf-8"))
    return h.hexdigest()


def _atomic_write_json(payload: dict, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.sort_index().to_parquet(tmp)
    os.replace(tmp, path)


def build_rolling_signature(y: pd.Series, X: pd.DataFrame) -> tuple[str, dict]:
    ref_end = pd.Timestamp(DATA_FIM_ROLLING)
    y_ref = y.loc[:ref_end]
    X_ref = X.loc[y_ref.index]
    payload = {
        "pipeline_version": "rolling-causal-t1-r1-2026-08-05",
        "protocol": "rolling_expanding_refit_168h_one_step_observed_state_update",
        "target_index_semantics": "tau=t+1",
        "information_set": "X_tau_minus_1,Q_through_tau_minus_1",
        "data_start": DATA_INICIO_ESTUDO,
        "rolling_start": DATA_INICIO_ROLLING,
        "rolling_end": DATA_FIM_ROLLING,
        "block_hours": ROLLING_BLOCK_HOURS,
        "sarimax_order": SARIMAX_ORDER,
        "sarimax_seasonal_order": SARIMAX_SEASONAL_ORDER,
        "exog_columns": list(map(str, X.columns)),
        "data_fingerprint": _data_fingerprint(y_ref, X_ref),
        "statsmodels_version": __import__("statsmodels").__version__,
    }
    return _json_hash(payload), payload


def audit_filter_does_not_use_current_target(
    fitted_result,
    y_block: pd.Series,
    X_block: pd.DataFrame,
    label: str,
) -> dict:
    """
    Altera artificialmente Q_tau e exige que a previsão emitida para o próprio
    tau permaneça idêntica. Alterações podem afetar tau+1 em diante, nunca tau.
    """
    n_probe = min(len(y_block), 72)
    y_probe = y_block.iloc[:n_probe].astype(float).copy()
    X_probe = X_block.iloc[:n_probe].astype(float).copy()
    base = pd.Series(
        np.asarray(
            fitted_result.extend(endog=y_probe, exog=X_probe).fittedvalues,
            dtype=float,
        ).reshape(-1),
        index=y_probe.index,
    )

    positions = sorted(set([0, min(12, n_probe - 1), min(36, n_probe - 1)]))
    max_same_time_difference = 0.0
    for pos in positions:
        altered = y_probe.copy()
        altered.iloc[pos] += 10000.0
        alt_pred = pd.Series(
            np.asarray(
                fitted_result.extend(endog=altered, exog=X_probe).fittedvalues,
                dtype=float,
            ).reshape(-1),
            index=y_probe.index,
        )
        diff = float(np.max(np.abs(base.iloc[: pos + 1] - alt_pred.iloc[: pos + 1])))
        max_same_time_difference = max(max_same_time_difference, diff)
        if not np.allclose(
            base.iloc[: pos + 1], alt_pred.iloc[: pos + 1],
            rtol=1e-10, atol=1e-7,
        ):
            raise AssertionError(
                f"AUDITORIA CAUSAL FALHOU ({label}): Q_tau alterou sua própria "
                "previsão ou uma previsão anterior."
            )

    print(
        f"AUDITORIA CAUSAL APROVADA ({label}): Q_tau não participa de "
        f"Qhat_tau|tau-1; diferença máxima={max_same_time_difference:.3e}"
    )
    return {
        "label": label,
        "passed": True,
        "probe_hours": n_probe,
        "positions_tested": positions,
        "max_same_time_difference": max_same_time_difference,
    }


def generate_or_resume_rolling_oos_predictions(
    y: pd.Series,
    X: pd.DataFrame,
    checkpoint_dir: Path,
    force_rebuild: bool = False,
) -> tuple[pd.Series, dict, Path, Path]:
    """
    Rolling OOS causal de 2023.

    Antes de cada bloco, os parâmetros são reestimados com janela expansiva
    terminando em block_start-1. Dentro do bloco, ``extend`` executa filtragem
    sequencial: fittedvalues[tau] é calculado com o estado até tau-1 e somente
    depois Q_tau é assimilado. O vetor do bloco é matematicamente equivalente
    ao laço get_forecast(1)+append, mas evita crescimento de memória por hora.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    signature, signature_payload = build_rolling_signature(y, X)
    tag = signature[:12]
    pred_path = checkpoint_dir / f"oos_2023_rolling_causal_T1_{tag}.parquet"
    meta_path = checkpoint_dir / f"oos_2023_rolling_causal_T1_{tag}.json"

    expected_index = pd.date_range(DATA_INICIO_ROLLING, DATA_FIM_ROLLING, freq="h")
    expected_hours = len(expected_index)
    pred_table = pd.DataFrame(index=expected_index)
    pred_table["pred"] = np.nan
    pred_table["block_num"] = pd.Series(pd.NA, index=expected_index, dtype="Int64")
    pred_table["issue_time"] = pd.NaT
    pred_table["fit_end"] = pd.NaT
    pred_table.index.name = "datetime"

    meta = {
        "signature": signature,
        "signature_payload": signature_payload,
        "expected_hours": expected_hours,
        "completed_blocks": [],
        "block_records": [],
        "created_at": datetime.now().isoformat(),
    }

    if force_rebuild:
        print("FORCE_REBUILD_ROLLING=True: checkpoint compatível será ignorado.")
    elif pred_path.exists() or meta_path.exists():
        if not (pred_path.exists() and meta_path.exists()):
            raise RuntimeError(
                "Checkpoint incompleto no Drive (Parquet ou JSON ausente). "
                f"Preserve os arquivos e use FORCE_REBUILD_ROLLING=True: {pred_path.parent}"
            )
        with meta_path.open("r", encoding="utf-8") as f:
            old_meta = json.load(f)
        if old_meta.get("signature") != signature:
            raise RuntimeError(
                "Checkpoint incompatível detectado. Ele não será reutilizado. "
                "Como o nome contém a assinatura, verifique se os arquivos foram "
                "renomeados manualmente."
            )
        previous = pd.read_parquet(pred_path)
        previous.index = pd.to_datetime(previous.index)
        if previous.index.has_duplicates:
            raise RuntimeError("Checkpoint possui timestamps duplicados.")
        common = expected_index.intersection(previous.index)
        pred_table.loc[common, previous.columns] = previous.reindex(common)
        meta = old_meta
        print(
            f"Checkpoint compatível encontrado: {pred_table['pred'].notna().sum()}/"
            f"{expected_hours} horas já calculadas."
        )

    blocks = []
    cursor = expected_index[0]
    while cursor <= expected_index[-1]:
        block_end = min(
            cursor + pd.Timedelta(hours=ROLLING_BLOCK_HOURS - 1),
            expected_index[-1],
        )
        blocks.append((cursor, block_end))
        cursor = block_end + pd.Timedelta(hours=1)

    audit_2023 = None
    for block_num, (block_start, block_end) in enumerate(blocks, start=1):
        block_index = pd.date_range(block_start, block_end, freq="h")
        existing = pd.to_numeric(
            pred_table.loc[block_index, "pred"], errors="coerce"
        )
        if existing.notna().all():
            print(
                f"[ROLLING {block_num:02d}/{len(blocks)}] CHECKPOINT OK — "
                f"{block_start} -> {block_end}",
                flush=True,
            )
            continue

        # Um bloco parcial nunca é misturado com um novo ajuste: ele é refeito.
        pred_table.loc[block_index, "pred"] = np.nan
        pred_table.loc[block_index, "block_num"] = pd.NA
        pred_table.loc[block_index, "issue_time"] = pd.NaT
        pred_table.loc[block_index, "fit_end"] = pd.NaT
        train_index = y.index[y.index < block_start]
        y_fit = y.loc[train_index]
        X_fit = X.loc[train_index]
        y_block = y.loc[block_index]
        X_block = X.loc[block_index]

        if y_fit.empty or len(y_fit) != len(X_fit):
            raise RuntimeError(f"Treino inválido antes do bloco {block_num}.")
        if y_fit.index.max() >= block_start:
            raise AssertionError("Vazamento: treino alcançou o bloco OOS.")
        if not y_block.index.equals(X_block.index):
            raise AssertionError("Índices de alvo e exógenas não coincidem no bloco.")
        if y_fit.isna().any() or X_fit.isna().any().any():
            raise ValueError(f"NaN no treino do bloco {block_num}.")
        if y_block.isna().any() or X_block.isna().any().any():
            raise ValueError(f"NaN no bloco OOS {block_num}.")

        print(
            f"[ROLLING {block_num:02d}/{len(blocks)}] AJUSTANDO — treino "
            f"{y_fit.index.min()} -> {y_fit.index.max()} ({len(y_fit)} h); "
            f"OOS {block_start} -> {block_end}",
            flush=True,
        )
        t0 = time.time()
        result = fit_sarimax(y_fit, X_fit)
        converged = bool(getattr(result, "mle_retvals", {}).get("converged", True))

        if audit_2023 is None:
            audit_2023 = audit_filter_does_not_use_current_target(
                result, y_block, X_block, label="primeiro bloco OOS de 2023"
            )

        filtered_block = result.extend(endog=y_block, exog=X_block)
        block_pred = pd.Series(
            np.asarray(filtered_block.fittedvalues, dtype=float).reshape(-1),
            index=block_index,
            name="pred",
        )
        if block_pred.isna().any() or len(block_pred) != len(block_index):
            raise RuntimeError(f"Previsões inválidas no bloco {block_num}.")

        pred_table.loc[block_index, "pred"] = block_pred.values
        pred_table.loc[block_index, "block_num"] = block_num
        pred_table.loc[block_index, "issue_time"] = block_index - pd.Timedelta(hours=1)
        pred_table.loc[block_index, "fit_end"] = y_fit.index.max()

        completed = []
        for bnum, (bstart, bend) in enumerate(blocks, start=1):
            bidx = pd.date_range(bstart, bend, freq="h")
            if pd.to_numeric(pred_table.loc[bidx, "pred"], errors="coerce").notna().all():
                completed.append(bnum)

        elapsed = time.time() - t0
        record = {
            "block_num": block_num,
            "train_start": y_fit.index.min(),
            "train_end": y_fit.index.max(),
            "forecast_start": block_start,
            "forecast_end": block_end,
            "n_train": len(y_fit),
            "n_forecast": len(block_index),
            "optimizer_converged": converged,
            "elapsed_seconds": elapsed,
            "saved_at": datetime.now().isoformat(),
        }
        prior_records = [
            r for r in meta.get("block_records", [])
            if int(r.get("block_num", -1)) != block_num
        ]
        meta["block_records"] = prior_records + [record]
        meta["completed_blocks"] = completed
        meta["last_completed_end"] = block_end
        meta["completed_hours"] = int(pred_table["pred"].notna().sum())
        meta["updated_at"] = datetime.now().isoformat()
        meta["audit_2023"] = audit_2023

        save_frame = pred_table.loc[pred_table["pred"].notna()].copy()
        _atomic_write_parquet(save_frame, pred_path)
        _atomic_write_json(meta, meta_path)
        print(
            f"[ROLLING {block_num:02d}/{len(blocks)}] SALVO — "
            f"{meta['completed_hours']}/{expected_hours} h | "
            f"{elapsed/60:.1f} min | convergiu={converged}",
            flush=True,
        )
        del result, filtered_block, block_pred, y_fit, X_fit, y_block, X_block
        gc.collect()

    pred = pd.to_numeric(pred_table["pred"], errors="coerce")
    if pred.isna().any():
        raise RuntimeError(
            f"Rolling incompleto: {int(pred.isna().sum())} das {expected_hours} h ausentes."
        )
    if len(pred) != 8760 or not pred.index.equals(expected_index):
        raise AssertionError("Rolling final não corresponde às 8.760 horas de 2023.")

    meta["complete"] = True
    meta["completed_hours"] = expected_hours
    meta["completed_at"] = datetime.now().isoformat()
    _atomic_write_json(meta, meta_path)
    pred.name = "pred"
    return pred.astype(float), meta, pred_path, meta_path



def load_existing_completed_rolling_strict(
    y: pd.Series,
    X: pd.DataFrame,
    pred_path: Path,
    meta_path: Path,
    expected_tag: str,
) -> tuple[pd.Series, dict, Path, Path]:
    """Carrega somente o rolling causal já concluído; nunca o recalcula."""
    pred_path = Path(pred_path)
    meta_path = Path(meta_path)

    if not pred_path.exists():
        raise FileNotFoundError(
            "Checkpoint rolling final não encontrado. O código NÃO irá recalculá-lo.\n"
            f"Esperado: {pred_path}"
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            "JSON do checkpoint rolling final não encontrado. O código NÃO irá recalculá-lo.\n"
            f"Esperado: {meta_path}"
        )

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    saved_signature = str(meta.get("signature", ""))
    if not saved_signature:
        raise RuntimeError("O JSON do rolling não contém 'signature'.")
    if saved_signature[:12] != str(expected_tag):
        raise RuntimeError(
            "O checkpoint encontrado não é o rolling final esperado. "
            f"tag JSON={saved_signature[:12]} | tag esperada={expected_tag}."
        )

    # Confirma que base, exógenas, protocolo e configuração são os mesmos do checkpoint.
    current_signature, _ = build_rolling_signature(y, X)
    if current_signature != saved_signature:
        raise RuntimeError(
            "O rolling salvo existe, mas sua assinatura não corresponde à base/configuração "
            "deste run. Por segurança, o código NÃO recalculará e NÃO misturará resultados.\n"
            f"salvo={saved_signature[:12]} | atual={current_signature[:12]}\n"
            f"Checkpoint: {pred_path}"
        )

    if not bool(meta.get("complete", False)):
        raise RuntimeError("O JSON informa que o rolling salvo não está completo.")
    if int(meta.get("completed_hours", -1)) != 8760:
        raise RuntimeError(
            f"Rolling salvo não possui 8760 h completas: {meta.get('completed_hours')}"
        )

    frame = pd.read_parquet(pred_path)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()

    if frame.index.has_duplicates:
        raise RuntimeError("Checkpoint rolling possui timestamps duplicados.")
    if "pred" not in frame.columns:
        raise RuntimeError("Checkpoint rolling não possui a coluna 'pred'.")

    expected_index = pd.date_range(DATA_INICIO_ROLLING, DATA_FIM_ROLLING, freq="h")
    if len(frame) != 8760 or not frame.index.equals(expected_index):
        raise RuntimeError(
            "Checkpoint rolling não corresponde exatamente às 8.760 horas de 2023."
        )

    pred = pd.to_numeric(frame["pred"], errors="coerce")
    if pred.isna().any():
        raise RuntimeError(
            f"Checkpoint rolling contém {int(pred.isna().sum())} previsões ausentes."
        )

    if "issue_time" in frame.columns:
        issue = pd.to_datetime(frame["issue_time"])
        expected_issue = frame.index - pd.Timedelta(hours=1)
        if not np.array_equal(issue.to_numpy(), expected_issue.to_numpy()):
            raise AssertionError("issue_time do rolling não corresponde a tau-1.")

    if "fit_end" in frame.columns and "block_num" in frame.columns:
        fit_end = pd.to_datetime(frame["fit_end"])
        block_num = pd.Series(frame["block_num"], index=frame.index).astype("Int64")
        for b in block_num.dropna().unique():
            mask = block_num == int(b)
            block_index = frame.index[mask.to_numpy()]
            fit_values = fit_end.loc[block_index].dropna()
            if len(block_index) and len(fit_values) and fit_values.max() >= block_index.min():
                raise AssertionError(
                    f"fit_end invade o bloco OOS {int(b)} no checkpoint salvo."
                )

    pred.name = "pred"
    print("\n" + "=" * 100)
    print("ROLLING CAUSAL EXISTENTE CARREGADO — NENHUM BLOCO SERÁ REFEITO")
    print("=" * 100)
    print(f"Checkpoint: {pred_path}")
    print(f"Assinatura: {saved_signature[:12]}")
    print(f"Horas OOS: {len(pred)}")
    print(f"Protocolo: {meta.get('signature_payload', {}).get('protocol')}")
    print("Compatibilidade base/configuração: APROVADA")
    print("=" * 100)

    return pred.astype(float), meta, pred_path, meta_path

def load_lstm_prediction(path: Path, y_reference: pd.Series) -> tuple[pd.Series | None, str | None]:
    """Carrega a série LSTM e exige o mesmo alvo/timestamps usados aqui."""
    path = Path(path)
    if not path.exists():
        print(f"AVISO: arquivo LSTM não encontrado: {path}")
        return None, None
    xls = pd.ExcelFile(path)
    sheet = "series_2024" if "series_2024" in xls.sheet_names else xls.sheet_names[0]
    frame = pd.read_excel(path, sheet_name=sheet)
    dt_col = next((c for c in ["datetime", "Data", "timestamp"] if c in frame.columns), None)
    if dt_col is None:
        raise ValueError(f"Data/hora não identificada no arquivo LSTM: {path}")
    frame[dt_col] = pd.to_datetime(frame[dt_col])
    frame = frame.set_index(dt_col).sort_index()
    pred_col = next(
        (c for c in frame.columns if "lstm" in str(c).lower() and "resid" not in str(c).lower()),
        None,
    )
    obs_col = next(
        (c for c in ["observed", "y_real", "Vazão", "vazao"] if c in frame.columns),
        None,
    )
    if pred_col is None:
        raise ValueError(f"Coluna de previsão LSTM não identificada em {path}.")
    pred = pd.to_numeric(frame[pred_col], errors="coerce").reindex(y_reference.index)
    if pred.isna().any():
        raise ValueError(
            f"A série LSTM não cobre os mesmos timestamps de 2024: "
            f"{int(pred.isna().sum())} valores ausentes."
        )
    if obs_col is not None:
        observed_file = pd.to_numeric(frame[obs_col], errors="coerce").reindex(y_reference.index)
        if not np.allclose(
            observed_file.to_numpy(dtype=float),
            y_reference.to_numpy(dtype=float),
            rtol=1e-8, atol=1e-5, equal_nan=False,
        ):
            raise AssertionError(
                "Comparação inválida: o arquivo LSTM usa uma série observada "
                "diferente da série-alvo deste pipeline."
            )
    pred.name = "pred_lstm"
    print(f"LSTM carregada e alinhada: {path} | coluna={pred_col} | {len(pred)} h")
    return pred.astype(float), str(path)

def build_post_rain_event_flag(
    X: pd.DataFrame,
    precip_col: str = "Precipitação",
    janela_horas: int = 14
) -> pd.Series:
    """
    X já está alinhado causalmente: na linha-alvo tau, Precipitação representa
    P_{tau-1}. A janela de 14 h, portanto, termina no instante de emissão.
    """
    if precip_col not in X.columns:
        raise ValueError(f"Coluna '{precip_col}' não encontrada.")
    rain_now = (X[precip_col] > 0).astype(int)
    return rain_now.rolling(window=janela_horas, min_periods=1).max().astype(bool)

def get_xgb_whitelist_columns(all_columns: list[str]) -> list[str]:
    exact_candidates = [
        "Precipitação",
        "precip_lag_1h",
        "precip_lag_2h",
        "precip_acum_20d",
        "interacao_chuva_saturacao",
        "tempo_sem_chuva",
        "Temperatura Média",
        "Temperatura Instantanea",
        "Umidade Instantanea",
        "Umidade Media",
        "Sensação Termica (°F)",
        "hora_sin",
        "hora_cos",
        "Classe do dia_Feriado",
    ]

    optional_exact = [
        "interacao_classe_periodo_Dia comum_Manha",
        "interacao_classe_periodo_Feriado_Manha",
        "interacao_classe_periodo_Feriado_Noite",
        "Dia da Semana_segunda-feira",
        "Dia da Semana_quarta-feira",
        "Dia da Semana_sábado",
    ]

    keep = []
    for c in exact_candidates + optional_exact:
        if c in all_columns:
            keep.append(c)

    return sorted(list(set(keep)))

def clip_series_by_train_quantiles(
    s: pd.Series,
    train_ref: pd.Series,
    q_inf: float = 0.01,
    q_sup: float = 0.99
) -> pd.Series:
    lo = train_ref.quantile(q_inf)
    hi = train_ref.quantile(q_sup)
    return s.clip(lower=lo, upper=hi)

def add_xgb_features(
    X_base: pd.DataFrame,
    pred_base: pd.Series,
    y_real: pd.Series | None = None,
    resid_hist_source: pd.Series | None = None
) -> pd.DataFrame:
    """
    Features do XGBoost no protocolo operacional ONLINE_1H_LAG24.

    Regras implementadas:
      1) Para cada alvo tau, vazão/resíduo entram apenas como lags: tau-1..tau-24.
      2) Nunca usa Q_tau ou resid_tau como feature.
      3) X_base já está alinhado: Precipitação na linha tau representa P_{tau-1}.
         As janelas pluviométricas terminam, portanto, no instante de emissão.
      4) Neste teste, as colunas precip_acum_* são substituídas por MÉDIAS prévias.
      5) São criados estados de erro recente para alimentar tanto o XGB quanto
         a matriz lambda contextual.
    """
    X = X_base.copy()
    pred_base = pd.Series(pred_base).reindex(X.index)
    X["pred_sarimax"] = pred_base

    # ------------------------------------------------------------
    # 1) Precipitação causal: na linha tau, P_available = P_{tau-1}
    # ------------------------------------------------------------
    if "Precipitação" in X.columns:
        P_available = X["Precipitação"].astype(float)
        P_hist = P_available

        X["flag_chuva"] = (P_available > 0).astype(int)
        X["precip_prev_1h"] = P_available

        # Médias prévias de chuva.
        X["precip_media_3h"] = P_hist.rolling(3, min_periods=1).mean()
        X["precip_media_6h"] = P_hist.rolling(6, min_periods=1).mean()
        X["precip_media_12h"] = P_hist.rolling(12, min_periods=1).mean()
        X["precip_media_24h"] = P_hist.rolling(24, min_periods=1).mean()

        # Somas temporárias: não são anexadas ao DataFrame, evitando oito
        # colunas diagnósticas duplicadas nas matrizes de treino e teste.
        precip_soma_3h = P_hist.rolling(3, min_periods=1).sum()
        precip_soma_6h = P_hist.rolling(6, min_periods=1).sum()
        precip_soma_12h = P_hist.rolling(12, min_periods=1).sum()
        precip_soma_24h = P_hist.rolling(24, min_periods=1).sum()

        # Substituição propriamente dita: o restante do código pode continuar usando
        # precip_acum_* como antes, mas agora o conteúdo é média, não acumulado.
        if SUBSTITUIR_ACUMULADOS_POR_MEDIAS:
            X["precip_acum_3h"] = X["precip_media_3h"]
            X["precip_acum_6h"] = X["precip_media_6h"]
            X["precip_acum_12h"] = X["precip_media_12h"]
            X["precip_acum_24h"] = X["precip_media_24h"]
        else:
            X["precip_acum_3h"] = precip_soma_3h
            X["precip_acum_6h"] = precip_soma_6h
            X["precip_acum_12h"] = precip_soma_12h
            X["precip_acum_24h"] = precip_soma_24h

        X["precip_max_6h"] = P_hist.rolling(6, min_periods=1).max()
        X["precip_max_12h"] = P_hist.rolling(12, min_periods=1).max()
        X["precip_max_24h"] = P_hist.rolling(24, min_periods=1).max()

        wet = (P_hist > 0).astype(float)
        X["precip_horas_chuva_6h"] = wet.rolling(6, min_periods=1).sum()
        X["precip_horas_chuva_24h"] = wet.rolling(24, min_periods=1).sum()

        X["precip_intensidade_chuvosa_6h"] = precip_soma_6h / (X["precip_horas_chuva_6h"] + 1e-6)
        X["precip_intensidade_chuvosa_24h"] = precip_soma_24h / (X["precip_horas_chuva_24h"] + 1e-6)

        for janela in [6, 12, 24]:
            decay = pd.Series(0.0, index=X.index)
            for lag in range(janela):
                decay = decay.add(
                    (ALPHA_PRECIP_DECAY ** lag) * P_available.shift(lag),
                    fill_value=0.0,
                )
            X[f"precip_decay_{janela}h"] = decay

        del precip_soma_3h, precip_soma_6h, precip_soma_12h, precip_soma_24h

    if "precip_acum_20d" in X.columns:
        if SUBSTITUIR_ACUM20D_POR_MEDIA20D:
            X["precip_media_20d"] = X["precip_acum_20d"] / float(HORAS_20D)
            X["precip_acum_20d"] = X["precip_media_20d"]

    # Se existir uma interação pré-processada que usava acumulado bruto, sobrescreve
    # para garantir que ela também use a nova média de 20 dias.
    if "Precipitação" in X.columns and "precip_acum_20d" in X.columns and "interacao_chuva_saturacao" in X.columns:
        X["interacao_chuva_saturacao"] = X["Precipitação"] * X["precip_acum_20d"]

    if "precip_lag_1h" in X.columns and "precip_lag_2h" in X.columns:
        X["chuva_curta"] = X["precip_lag_1h"] + X["precip_lag_2h"]
    elif "Precipitação" in X.columns and "precip_lag_2h" in X.columns:
        X["chuva_curta"] = X["Precipitação"] + X["precip_lag_2h"]

    if "tempo_sem_chuva" in X.columns and "Temperatura Média" in X.columns:
        X["inter_seca_temperatura"] = np.log1p(X["tempo_sem_chuva"].clip(lower=0)) * X["Temperatura Média"]
        if "Precipitação" in X.columns:
            X["inter_seca_temperatura"] = (
                X["inter_seca_temperatura"] * (X["Precipitação"] == 0).astype(int)
            )

    if "tempo_sem_chuva" in X.columns and "precip_acum_20d" in X.columns:
        X["inter_seca_saturacao"] = X["tempo_sem_chuva"] * X["precip_acum_20d"]

    if "Precipitação" in X.columns and "precip_acum_20d" in X.columns:
        X["inter_precip20d"] = X["Precipitação"] * X["precip_acum_20d"]

    if "interacao_chuva_saturacao" in X.columns and "hora_sin" in X.columns:
        X["inter_sat_hora"] = X["interacao_chuva_saturacao"] * X["hora_sin"]

    X["pred_sarimax_lag_1"] = X["pred_sarimax"].shift(1)
    X["pred_sarimax_lag_2"] = X["pred_sarimax"].shift(2)
    X["pred_sarimax_lag_24"] = X["pred_sarimax"].shift(24)
    X["delta_pred_1h"] = X["pred_sarimax"] - X["pred_sarimax_lag_1"]
    X["delta_pred_3h"] = X["pred_sarimax"] - X["pred_sarimax"].shift(3)

    # ------------------------------------------------------------
    # 3) Protocolo ONLINE 1H: caixa temporal defasada
    # ------------------------------------------------------------
    # Para cada instante alvo tau = X.index[i], usa somente dados observados
    # até tau-1. Isto é causal para previsão 1 passo à frente/online.
    idx = pd.DatetimeIndex(X.index)

    X["online_horizonte_h"] = 1.0
    # Mantém nomes daily_* para compatibilidade com a matriz de estado já existente.
    X["daily_update_hour"] = -1
    X["daily_horizonte_h"] = 1.0
    X["daily_horizonte_sin"] = np.sin(2 * np.pi * X["daily_horizonte_h"] / 24.0)
    X["daily_horizonte_cos"] = np.cos(2 * np.pi * X["daily_horizonte_h"] / 24.0)

    if y_real is not None:
        y_source = pd.Series(y_real).sort_index()
        y_source = y_source[~y_source.index.duplicated(keep="last")]

        # Q_{tau-1}, Q_{tau-2}, ..., Q_{tau-24}
        for lag in range(1, TAMANHO_PACOTE_VAZAO_HORAS + 1):
            X[f"vazao_lag_{lag}"] = y_source.shift(lag).reindex(X.index).to_numpy()

        vazao_cols = [f"vazao_lag_{lag}" for lag in range(1, TAMANHO_PACOTE_VAZAO_HORAS + 1)]
        X["vazao_media_24h_pacote"] = X[vazao_cols].mean(axis=1)
        X["vazao_max_24h_pacote"] = X[vazao_cols].max(axis=1)
        X["vazao_min_24h_pacote"] = X[vazao_cols].min(axis=1)
        X["vazao_std_24h_pacote"] = X[vazao_cols].std(axis=1)

        X["delta_vazao_1h"] = X["vazao_lag_1"] - X["vazao_lag_2"]
        X["delta_vazao_3h"] = X["vazao_lag_1"] - X["vazao_lag_4"]
        X["delta_vazao_6h"] = X["vazao_lag_1"] - X["vazao_lag_7"]
        X["delta_vazao_24h"] = X["vazao_lag_1"] - X["vazao_lag_24"]

        X["atividade_hidraulica_24h"] = (
            (X["delta_vazao_24h"].abs() / (X["vazao_media_24h_pacote"].abs() + 1.0)) +
            (X["vazao_std_24h_pacote"] / (X["vazao_media_24h_pacote"].abs() + 1.0))
        )

        if "tempo_sem_chuva" in X.columns:
            X["tempo_sem_chuva_ajustado"] = X["tempo_sem_chuva"] / (1.0 + 3.0 * X["atividade_hidraulica_24h"].clip(lower=0))
            if "Temperatura Média" in X.columns:
                X["inter_seca_temp_ajustada"] = np.log1p(X["tempo_sem_chuva_ajustado"].clip(lower=0)) * X["Temperatura Média"]
                if "Precipitação" in X.columns:
                    X["inter_seca_temp_ajustada"] *= (X["Precipitação"] == 0).astype(int)

        X["gap_pred_vazao"] = X["pred_sarimax"] - X["vazao_lag_1"]
        X["gap_pred_vazao_media24"] = X["pred_sarimax"] - X["vazao_media_24h_pacote"]

    if resid_hist_source is not None:
        resid_full = pd.Series(resid_hist_source).sort_index()
        resid_full = resid_full[~resid_full.index.duplicated(keep="last")]

        # e_{tau-1}, e_{tau-2}, ..., e_{tau-24}; por padrão, e é erro do SARIMAX.
        for lag in range(1, TAMANHO_PACOTE_VAZAO_HORAS + 1):
            X[f"resid_lag_{lag}"] = resid_full.shift(lag).reindex(X.index).to_numpy()

        resid_cols = [f"resid_lag_{lag}" for lag in range(1, TAMANHO_PACOTE_VAZAO_HORAS + 1)]
        X["abs_resid_lag_1"] = X["resid_lag_1"].abs()
        X["resid_media_24h_pacote"] = X[resid_cols].mean(axis=1)
        X["abs_resid_media_24h_pacote"] = X[resid_cols].abs().mean(axis=1)
        X["resid_max_24h_pacote"] = X[resid_cols].max(axis=1)
        X["resid_min_24h_pacote"] = X[resid_cols].min(axis=1)
        X["super_lag_1_mag"] = np.maximum(-X["resid_lag_1"], 0)
        X["sub_lag_1_mag"] = np.maximum(X["resid_lag_1"], 0)
        X["super_memoria_24h"] = X[resid_cols].clip(upper=0).abs().mean(axis=1)
        X["sub_memoria_24h"] = X[resid_cols].clip(lower=0).mean(axis=1)
        X["n_super_24h_pacote"] = (X[resid_cols] < -150).sum(axis=1)
        X["n_sub_24h_pacote"] = (X[resid_cols] > 150).sum(axis=1)

        # Esta coluna textual é usada apenas pela matriz de estado/lambda.
        # Ela NÃO deve entrar no XGBoost como feature bruta, pois XGBoost exige numérico.
        X["erro_sinal_24h"] = "neutro"
        X.loc[X["n_sub_24h_pacote"] > X["n_super_24h_pacote"], "erro_sinal_24h"] = "sub"
        X.loc[X["n_super_24h_pacote"] > X["n_sub_24h_pacote"], "erro_sinal_24h"] = "super"

        # Versões numéricas equivalentes, permitidas para o XGBoost.
        X["erro_sinal_sub_24h"] = (X["erro_sinal_24h"] == "sub").astype(int)
        X["erro_sinal_super_24h"] = (X["erro_sinal_24h"] == "super").astype(int)
        X["erro_sinal_neutro_24h"] = (X["erro_sinal_24h"] == "neutro").astype(int)

    return X


def select_top_features_by_gain(model, X_train: pd.DataFrame, top_k: int = 15) -> list[str]:
    gains = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return gains.head(top_k).index.tolist()

# ============================================================
# 3. FUNÇÕES DA MATRIZ DE ESTADO x_t E LAMBDA
# ============================================================

def _safe_cut(s: pd.Series, bins, default="NA") -> pd.Series:
    if s is None:
        return pd.Series(default)
    try:
        return pd.cut(s.astype(float), bins=bins, include_lowest=True).astype(str)
    except Exception:
        return pd.Series(default, index=s.index, dtype="object")


def _safe_qcut(s: pd.Series, q: int = 5, default="QNA") -> pd.Series:
    try:
        return pd.qcut(s.astype(float), q=q, duplicates="drop").astype(str)
    except Exception:
        return pd.Series(default, index=s.index, dtype="object")


def fit_quantile_bins(s: pd.Series, q: int = 5) -> np.ndarray:
    values = pd.Series(s).dropna().astype(float)
    if values.empty:
        return np.array([-np.inf, np.inf], dtype=float)
    edges = np.unique(values.quantile(np.linspace(0, 1, q + 1)).to_numpy(dtype=float))
    if len(edges) < 2:
        return np.array([-np.inf, np.inf], dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def build_state_dataframe(
    df_feat: pd.DataFrame,
    pred_bins: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Matriz de estado x_t para lambda contextual.

    O estado completo combina chuva média, pico de chuva, persistência de chuva,
    tempo sem chuva ajustado, erro recente do pacote 24h, gap SARIMAX-vazão,
    nível previsto pelo SARIMAX e horizonte dentro do bloco diário.
    """
    # Somente leitura; copiar todas as features aqui duplicava dezenas de MB.
    X = df_feat
    out = pd.DataFrame(index=X.index)

    pmean_bins = [-0.001, 0, 0.2, 1, 2.5, 5, 10, np.inf]
    pmax_bins = [-0.001, 0, 2.5, 5, 10, 20, 40, np.inf]
    wet_bins = [-0.001, 0, 1, 3, 6, 12, 24]
    tsc_bins = [-0.001, 0, 6, 24, 72, 168, 336, 720, np.inf]
    err_mean_bins = [-np.inf, -300, -150, -50, 50, 150, 300, np.inf]
    err_abs_bins = [-0.001, 50, 100, 150, 250, 400, 700, np.inf]
    gap_bins = [-np.inf, -600, -300, -100, 100, 300, 600, np.inf]
    horizonte_bins = [0, 6, 12, 18, 24]

    out["st_pmean6"] = _safe_cut(X["precip_media_6h"], pmean_bins) if "precip_media_6h" in X.columns else "NA"
    out["st_pmean24"] = _safe_cut(X["precip_media_24h"], pmean_bins) if "precip_media_24h" in X.columns else "NA"
    out["st_pmax6"] = _safe_cut(X["precip_max_6h"], pmax_bins) if "precip_max_6h" in X.columns else "NA"
    out["st_wet24"] = _safe_cut(X["precip_horas_chuva_24h"], wet_bins) if "precip_horas_chuva_24h" in X.columns else "NA"

    if "tempo_sem_chuva_ajustado" in X.columns:
        out["st_tsc_adj"] = _safe_cut(X["tempo_sem_chuva_ajustado"], tsc_bins)
    elif "tempo_sem_chuva" in X.columns:
        out["st_tsc_adj"] = _safe_cut(X["tempo_sem_chuva"], tsc_bins)
    else:
        out["st_tsc_adj"] = "NA"

    out["st_err_mean24"] = _safe_cut(X["resid_media_24h_pacote"], err_mean_bins) if "resid_media_24h_pacote" in X.columns else "NA"
    out["st_err_abs24"] = _safe_cut(X["abs_resid_media_24h_pacote"], err_abs_bins) if "abs_resid_media_24h_pacote" in X.columns else "NA"
    out["st_err_sign24"] = X["erro_sinal_24h"].astype(str) if "erro_sinal_24h" in X.columns else "NA"
    out["st_gap"] = _safe_cut(X["gap_pred_vazao"], gap_bins) if "gap_pred_vazao" in X.columns else "NA"
    if "pred_sarimax" in X.columns:
        if pred_bins is None:
            pred_bins = fit_quantile_bins(X["pred_sarimax"], q=5)
        out["st_pred"] = _safe_cut(X["pred_sarimax"], pred_bins)
    else:
        out["st_pred"] = "NA"
    out["st_horizonte"] = _safe_cut(X["daily_horizonte_h"], horizonte_bins) if "daily_horizonte_h" in X.columns else "NA"

    out["estado_xt"] = (
        "PM6=" + out["st_pmean6"] +
        "|PM24=" + out["st_pmean24"] +
        "|PX6=" + out["st_pmax6"] +
        "|W24=" + out["st_wet24"] +
        "|TSCa=" + out["st_tsc_adj"] +
        "|Emean=" + out["st_err_mean24"] +
        "|Eabs=" + out["st_err_abs24"] +
        "|Esign=" + out["st_err_sign24"] +
        "|G=" + out["st_gap"] +
        "|PS=" + out["st_pred"] +
        "|H=" + out["st_horizonte"]
    )

    out["estado_medio"] = (
        "PM24=" + out["st_pmean24"] +
        "|PX6=" + out["st_pmax6"] +
        "|TSCa=" + out["st_tsc_adj"] +
        "|Esign=" + out["st_err_sign24"] +
        "|G=" + out["st_gap"] +
        "|H=" + out["st_horizonte"]
    )

    out["estado_erro"] = (
        "Esign=" + out["st_err_sign24"] +
        "|Emean=" + out["st_err_mean24"] +
        "|Eabs=" + out["st_err_abs24"] +
        "|G=" + out["st_gap"] +
        "|H=" + out["st_horizonte"]
    )

    out["estado_chuva"] = (
        "PM24=" + out["st_pmean24"] +
        "|PX6=" + out["st_pmax6"] +
        "|W24=" + out["st_wet24"] +
        "|TSCa=" + out["st_tsc_adj"]
    )

    return out


def estimate_global_lambda(y_true_resid: pd.Series, y_pred_resid: pd.Series) -> float:
    y_true_resid = pd.Series(y_true_resid).astype(float)
    y_pred_resid = pd.Series(y_pred_resid).astype(float)
    num = np.sum(y_true_resid * y_pred_resid)
    den = np.sum(y_pred_resid ** 2)
    if den <= 1e-12:
        return LAMBDA_PRIOR_NEUTRO
    lam = num / den
    return float(np.clip(lam, LAMBDA_CLIP_MIN, LAMBDA_CLIP_MAX))


def _lambda_table_for_level(base: pd.DataFrame, level_col: str, min_obs: int, shrink_strength: int) -> pd.DataFrame:
    rows = []
    for estado, g in base.groupby(level_col):
        n = len(g)
        den = np.sum(g["y_pred_resid"] ** 2)
        if den <= 1e-12:
            lam_raw = LAMBDA_PRIOR_NEUTRO
        else:
            lam_raw = np.sum(g["y_pred_resid"] * g["y_true_resid"]) / den
        lam_raw = float(np.clip(lam_raw, LAMBDA_CLIP_MIN, LAMBDA_CLIP_MAX))

        true_sign = np.sign(g["y_true_resid"].values)
        pred_sign = np.sign(g["y_pred_resid"].values)
        nonzero = (true_sign != 0) & (pred_sign != 0)
        sign_acc = float((true_sign[nonzero] == pred_sign[nonzero]).mean()) if nonzero.sum() > 0 else np.nan

        if LAMBDA_AMORTECER_SE_SINAL_RUIM and np.isfinite(sign_acc) and sign_acc < LAMBDA_MIN_SIGN_ACC:
            lam_target = 0.0
        else:
            lam_target = lam_raw

        w = n / (n + shrink_strength)
        if n < min_obs:
            usable = False
            lam_final = LAMBDA_PRIOR_NEUTRO
        else:
            usable = True
            lam_final = w * lam_target + (1 - w) * LAMBDA_PRIOR_NEUTRO
        lam_final = float(np.clip(lam_final, LAMBDA_CLIP_MIN, LAMBDA_CLIP_MAX))

        rows.append({
            "level": level_col,
            "estado": str(estado),
            "n_obs": int(n),
            "min_obs": int(min_obs),
            "usable": bool(usable),
            "lambda_raw": lam_raw,
            "lambda_final": lam_final,
            "peso_estado": float(w),
            "sign_acc": sign_acc,
            "resid_mean": float(g["y_true_resid"].mean()),
            "abs_resid_mean": float(g["y_true_resid"].abs().mean()),
            "pred_resid_mean": float(g["y_pred_resid"].mean()),
            "rmse_resid_xgb": float(np.sqrt(np.mean((g["y_true_resid"] - g["y_pred_resid"]) ** 2))),
            "sub_pct": float((g["y_true_resid"] > 0).mean()),
            "super_pct": float((g["y_true_resid"] < 0).mean()),
        })
    return pd.DataFrame(rows)


def estimate_lambda_by_state(state_df: pd.DataFrame, y_true_resid: pd.Series, y_pred_resid: pd.Series,
                             min_obs: int = 25, shrink_strength: int = 40) -> pd.DataFrame:
    """
    Lambda contextual hierárquico.
    O lambda global é calculado apenas como diagnóstico, não como fallback dominante.
    O fallback é lambda=1.0, que preserva a correção do XGB sem amplificação.
    """
    base = state_df.copy()
    base["y_true_resid"] = pd.Series(y_true_resid).reindex(base.index).astype(float)
    base["y_pred_resid"] = pd.Series(y_pred_resid).reindex(base.index).astype(float)
    base = base.dropna(subset=["y_true_resid", "y_pred_resid"])

    lambda_global_raw = estimate_global_lambda(base["y_true_resid"], base["y_pred_resid"])
    tables = []
    level_plan = [
        ("estado_xt", LAMBDA_MIN_OBS_ESTADO_COMPLETO),
        ("estado_medio", LAMBDA_MIN_OBS_ESTADO_MEDIO),
        ("estado_erro", LAMBDA_MIN_OBS_ESTADO_ERRO),
        ("estado_chuva", LAMBDA_MIN_OBS_ESTADO_ERRO),
    ]
    for level_col, min_obs_level in level_plan:
        if level_col in base.columns:
            t = _lambda_table_for_level(base, level_col, min_obs_level, shrink_strength)
            if len(t) > 0:
                tables.append(t)
    lambda_df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    lambda_df.attrs["lambda_global_raw"] = lambda_global_raw
    lambda_df.attrs["lambda_fallback"] = LAMBDA_PRIOR_NEUTRO
    return lambda_df


def apply_lambda_by_state(state_df: pd.DataFrame, pred_resid_xgb: pd.Series, lambda_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pred_resid_xgb = pd.Series(pred_resid_xgb).astype(float)
    if len(lambda_df) and "usable" in lambda_df.columns:
        usable = lambda_df[lambda_df["usable"] == True].copy()
    else:
        usable = pd.DataFrame()

    lambda_maps = {}
    if len(usable):
        for level in usable["level"].unique():
            g = usable.loc[usable["level"] == level]
            lambda_maps[level] = dict(zip(g["estado"].astype(str), g["lambda_final"].astype(float)))

    levels = ["estado_xt", "estado_medio", "estado_erro", "estado_chuva"]
    lambda_vals, lambda_level = [], []
    for idx, row in state_df.reindex(pred_resid_xgb.index).iterrows():
        lam = LAMBDA_PRIOR_NEUTRO
        used = "fallback_1"
        for level in levels:
            if level in row.index and level in lambda_maps:
                key = str(row[level])
                if key in lambda_maps[level]:
                    lam = lambda_maps[level][key]
                    used = level
                    break
        lambda_vals.append(lam)
        lambda_level.append(used)

    lambda_t = pd.Series(lambda_vals, index=pred_resid_xgb.index, name="lambda_t")
    lambda_t.attrs["lambda_level"] = pd.Series(lambda_level, index=pred_resid_xgb.index, name="lambda_level")
    pred_corrigida = pred_resid_xgb * lambda_t
    return pred_corrigida, lambda_t


def diagnostico_estado_lambda(state_df: pd.DataFrame, y_true_resid: pd.Series, y_pred_resid: pd.Series,
                              lambda_df: pd.DataFrame, level: str = "estado_medio") -> pd.DataFrame:
    base = state_df.copy()
    base["y_true_resid"] = pd.Series(y_true_resid).reindex(base.index)
    base["y_pred_resid"] = pd.Series(y_pred_resid).reindex(base.index)
    base = base.dropna(subset=["y_true_resid", "y_pred_resid"])
    if level not in base.columns:
        return pd.DataFrame()
    rows = []
    for estado, g in base.groupby(level):
        resid = g["y_true_resid"]
        pred = g["y_pred_resid"]
        rows.append({
            "level": level,
            "estado": str(estado),
            "n": len(g),
            "resid_mean": float(resid.mean()),
            "resid_median": float(resid.median()),
            "abs_resid_mean": float(resid.abs().mean()),
            "rmse_xgb_resid": float(np.sqrt(np.mean((resid - pred) ** 2))),
            "pred_resid_mean": float(pred.mean()),
            "sub_pct": float((resid > 0).mean()),
            "super_pct": float((resid < 0).mean()),
            "sign_acc": float(((np.sign(resid) == np.sign(pred)) & (np.sign(resid) != 0) & (np.sign(pred) != 0)).mean()),
        })
    out = pd.DataFrame(rows).sort_values(["n", "abs_resid_mean"], ascending=[False, False])
    if len(lambda_df) and "level" in lambda_df.columns:
        lam = lambda_df[(lambda_df["level"] == level)][["estado", "lambda_final", "lambda_raw", "usable", "n_obs"]].copy()
        out = out.merge(lam, on="estado", how="left")
    return out


# ============================================================
# 4. CARREGAMENTO DOS DADOS
# ============================================================

if IN_COLAB and not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive", force_remount=False)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not Path(CAMINHO_ARQUIVO).exists():
    raise FileNotFoundError(
        f"Base tratada não encontrada: {CAMINHO_ARQUIVO}. "
        "Defina SARIMAX_DATA_PATH com o caminho correto."
    )

df = pd.read_excel(CAMINHO_ARQUIVO)

if COL_DATA not in df.columns:
    if "Data" in df.columns:
        COL_DATA = "Data"
    else:
        raise ValueError("Coluna de data não encontrada.")

df[COL_DATA] = pd.to_datetime(df[COL_DATA])
if df[COL_DATA].duplicated().any():
    duplicadas = int(df[COL_DATA].duplicated().sum())
    raise ValueError(f"A base contém {duplicadas} timestamps duplicados.")
df = df.sort_values(COL_DATA).set_index(COL_DATA)

df = df.loc[DATA_INICIO_ESTUDO:DATA_FIM_ESTUDO]
expected_full_index = pd.date_range(DATA_INICIO_ESTUDO, DATA_FIM_ESTUDO, freq="h")
if not df.index.equals(expected_full_index):
    missing_time = expected_full_index.difference(df.index)
    extra_time = df.index.difference(expected_full_index)
    raise ValueError(
        "A base tratada não possui uma grade horária completa. "
        f"Timestamps ausentes={len(missing_time)}; extras={len(extra_time)}. "
        "O código não fará preenchimento silencioso da série-alvo."
    )

y = df[TARGET_COL].copy()
X_raw = df.select_dtypes(include=[np.number]).drop(columns=[TARGET_COL]).copy()
if y.isna().any():
    raise ValueError(
        f"A série tratada ainda contém {int(y.isna().sum())} vazões ausentes."
    )
if X_raw.isna().any().any():
    missing_cols = X_raw.columns[X_raw.isna().any()].tolist()
    raise ValueError(
        "A matriz explicativa tratada ainda contém NaN nas colunas: "
        f"{missing_cols}."
    )
X = align_all_predictors_for_t_plus_1(X_raw)

# A primeira linha perde todas as variáveis explicativas após shift(1). Ela é
# removida sem preenchimento futuro. Nenhum bfill é permitido.
valid_index = X.index[y.notna() & X.notna().all(axis=1)]
df = df.loc[valid_index]
y = y.loc[valid_index]
X = X.loc[valid_index]

# Auditoria por amostra, sem construir uma segunda cópia integral de X.
audit_pos = np.unique(np.linspace(0, len(X) - 1, num=min(25, len(X)), dtype=int))
audit_target_idx = X.index[audit_pos]
audit_source_idx = audit_target_idx - pd.Timedelta(hours=1)
actual_audit = X.loc[audit_target_idx].to_numpy(dtype=np.float64)
expected_audit = X_raw.reindex(audit_source_idx).to_numpy(dtype=np.float64)
if not np.allclose(actual_audit, expected_audit, equal_nan=True):
    raise AssertionError("Falha no alinhamento t -> t+1 da matriz explicativa.")
del actual_audit, expected_audit, audit_target_idx, audit_source_idx, audit_pos, X_raw
gc.collect()

xgb_whitelist = get_xgb_whitelist_columns(list(X.columns))

mask_train_total = (df.index >= DATA_INICIO_ESTUDO) & (df.index < DATA_CORTE_TESTE)
mask_test = df.index >= DATA_CORTE_TESTE

y_train_total = y.loc[mask_train_total]
X_train_total = X.loc[mask_train_total]

y_test = y.loc[mask_test]
X_test = X.loc[mask_test]

print("Treino total:", y_train_total.index.min(), "->", y_train_total.index.max(), f"({len(y_train_total)} obs)")
print("Teste:", y_test.index.min(), "->", y_test.index.max(), f"({len(y_test)} obs)")
print(f"Todas as {X.shape[1]} variáveis explicativas foram deslocadas em 1 h.")
print(f"Protocolo causal: linha-alvo tau contém X_(tau-1), Q_(tau-1..tau-{TAMANHO_PACOTE_VAZAO_HORAS}) e resíduos disponíveis até tau-1.")
print(f"Janela pós-chuva: {JANELA_RESPOSTA_CHUVA_HORAS} horas")
print(f"Quantil hard cases: {QUANTIL_ERRO_DIFICIL}")
print(f"Teste chuva: acumulados substituídos por médias = {SUBSTITUIR_ACUMULADOS_POR_MEDIAS}")
print(f"Teste chuva: precip_acum_20d substituído por média 20d = {SUBSTITUIR_ACUM20D_POR_MEDIA20D}")

# ============================================================
# 5. CARREGAR O ROLLING OOS CAUSAL DE 2023 JÁ CONCLUÍDO
# ============================================================

print("\n[1/10] Carregando o rolling causal final já pronto — SEM recalcular 53 blocos...")
pred_oos_2023, rolling_meta, rolling_pred_path, rolling_meta_path = (
    load_existing_completed_rolling_strict(
        y=y,
        X=X,
        pred_path=EXISTING_ROLLING_PARQUET,
        meta_path=EXISTING_ROLLING_JSON,
        expected_tag=EXISTING_ROLLING_TAG,
    )
)

idx_oos_esperado = pd.date_range(DATA_INICIO_ROLLING, DATA_FIM_ROLLING, freq="h")
pred_oos_2023 = pred_oos_2023.reindex(idx_oos_esperado)
if pred_oos_2023.isna().any() or len(pred_oos_2023) != 8760:
    raise RuntimeError("O rolling causal de 2023 não foi concluído corretamente.")

y_oos_2023 = y.loc[pred_oos_2023.index]
resid_oos_2023 = y_oos_2023 - pred_oos_2023

print(f"Resíduos OOS 2023 concluídos: {len(resid_oos_2023)} observações")
print(f"Checkpoint Parquet: {rolling_pred_path}")
print(f"Checkpoint JSON:    {rolling_meta_path}")
print("Convenção validada: OOS indexado em tau; alvo Q_tau; emissão em tau-1.")

# ============================================================
# 6. SARIMAX FINAL
# ============================================================

print("\n[2/10] Treinando SARIMAX final em 2022-2023...")
sarimax_final = fit_sarimax(y_train_total, X_train_total)

audit_2024 = audit_filter_does_not_use_current_target(
    sarimax_final,
    y_test.iloc[:72],
    X_test.iloc[:72],
    label="aplicação independente de 2024",
)

print("[3/10] Prevendo 2024 com SARIMAX causal de um passo e armazenando estados para latência...")
sarimax_test_filtered = sarimax_final.extend(endog=y_test, exog=X_test)
pred_sarimax_2024 = pd.Series(
    np.asarray(sarimax_test_filtered.fittedvalues, dtype=np.float64).reshape(-1),
    index=y_test.index,
    name="pred_sarimax",
)
if pred_sarimax_2024.isna().any() or len(pred_sarimax_2024) != len(y_test):
    raise RuntimeError("Previsão SARIMAX causal de 2024 inválida.")

# Cache dos estados/matrizes para o experimento de latência; NÃO haverá segundo fit SARIMAX.
SARIMAX_LAT_TRAIN_STATES = pd.DataFrame(
    np.asarray(sarimax_final.filter_results.filtered_state, dtype=float).T,
    index=y_train_total.index,
)
SARIMAX_LAT_TEST_STATES = pd.DataFrame(
    np.asarray(sarimax_test_filtered.filter_results.filtered_state, dtype=float).T,
    index=y_test.index,
)
_ssm_latency = sarimax_test_filtered.model.ssm
SARIMAX_LAT_Z = np.asarray(_ssm_latency["design"], dtype=float).copy()
SARIMAX_LAT_T = np.asarray(_ssm_latency["transition"], dtype=float).copy()
SARIMAX_LAT_C = np.asarray(_ssm_latency["state_intercept"], dtype=float).copy()
SARIMAX_LAT_D = np.asarray(_ssm_latency["obs_intercept"], dtype=float).copy()

del sarimax_test_filtered, sarimax_final, _ssm_latency
gc.collect()

resid_sarimax_2024 = y_test - pred_sarimax_2024

# ============================================================
# 7. EVENTOS PÓS-CHUVA
# ============================================================

print("\n[4/10] Definindo janelas de resposta à chuva...")

event_mask_2023 = build_post_rain_event_flag(
    X.loc[pred_oos_2023.index],
    precip_col="Precipitação",
    janela_horas=JANELA_RESPOSTA_CHUVA_HORAS
)

event_mask_2024 = build_post_rain_event_flag(
    X_test,
    precip_col="Precipitação",
    janela_horas=JANELA_RESPOSTA_CHUVA_HORAS
)

print(f"Janelas de resposta em 2023: {event_mask_2023.sum()} de {len(event_mask_2023)} horas")
print(f"Janelas de resposta em 2024: {event_mask_2024.sum()} de {len(event_mask_2024)} horas")

# ============================================================
# 8. FEATURES XGB
# ============================================================

print("\n[5/10] Construindo features do XGBoost...")

X_xgb_train_base = X.loc[pred_oos_2023.index, xgb_whitelist].copy()
X_xgb_test_base = X_test[xgb_whitelist].copy()

# Fonte operacional de vazão e resíduos para o protocolo ONLINE_1H_LAG24.
# Em 2024, a função só acessa vazões/resíduos via shift(1..24), isto é, até tau-1.
y_operacional_full = y.loc[DATA_INICIO_ESTUDO:DATA_FIM_ESTUDO].copy()
resid_operacional_full = pd.concat([resid_oos_2023, resid_sarimax_2024]).sort_index()

X_xgb_train_full = add_xgb_features(
    X_base=X_xgb_train_base,
    pred_base=pred_oos_2023,
    y_real=y_operacional_full,
    resid_hist_source=resid_oos_2023
)
y_xgb_train_full = resid_oos_2023.copy()

X_xgb_test_full = add_xgb_features(
    X_base=X_xgb_test_base,
    pred_base=pred_sarimax_2024,
    y_real=y_operacional_full,
    resid_hist_source=resid_operacional_full
)

# Auditoria explícita dos dois preditores de memória mais sensíveis.
expected_q_lag1_2024 = y_operacional_full.shift(1).reindex(y_test.index)
if not np.allclose(
    X_xgb_test_full["vazao_lag_1"].to_numpy(dtype=float),
    expected_q_lag1_2024.to_numpy(dtype=float),
    equal_nan=True,
):
    raise AssertionError("Vazamento/alinhamento incorreto em vazao_lag_1 de 2024.")

expected_e_lag1_2024 = resid_operacional_full.shift(1).reindex(y_test.index)
if not np.allclose(
    X_xgb_test_full["resid_lag_1"].to_numpy(dtype=float),
    expected_e_lag1_2024.to_numpy(dtype=float),
    equal_nan=True,
):
    raise AssertionError("Vazamento/alinhamento incorreto em resid_lag_1 de 2024.")

for forbidden in [TARGET_COL, "y_real", "resid_sarimax"]:
    if forbidden in X_xgb_train_full.columns or forbidden in X_xgb_test_full.columns:
        raise AssertionError(f"Variável-alvo proibida encontrada no XGBoost: {forbidden}")
print("AUDITORIA XGBOOST APROVADA: Q e resíduos entram somente com lag >= 1.")

resid_event_2023 = resid_oos_2023.loc[event_mask_2023]
limiar_abs_resid = resid_event_2023.abs().quantile(QUANTIL_ERRO_DIFICIL)
hard_mask_2023 = event_mask_2023 & (resid_oos_2023.abs() >= limiar_abs_resid)

print(f"Limiar |resíduo| para hard cases: {limiar_abs_resid:.4f}")
print(f"Hard cases de treino: {hard_mask_2023.sum()} de {len(hard_mask_2023)} horas")

X_xgb_train = X_xgb_train_full.loc[hard_mask_2023].copy()
y_xgb_train = y_xgb_train_full.loc[hard_mask_2023].copy()

common_cols = sorted(set(X_xgb_train.columns).intersection(set(X_xgb_test_full.columns)))

# XGBoost não aceita colunas object/string. Mantemos essas colunas em X_xgb_train_full
# e X_xgb_test_full para a matriz de estado, mas removemos da matriz numérica do XGB.
common_cols_numeric = []
removed_non_numeric_cols = []
excluded_precip_diag_cols = []
for c in common_cols:
    # Exclui somas diagnósticas para que o teste seja realmente "média no lugar de acumulado".
    if c.startswith("precip_soma_") or c.endswith("_original_diag"):
        excluded_precip_diag_cols.append(c)
        continue
    is_train_num = pd.api.types.is_numeric_dtype(X_xgb_train[c]) or pd.api.types.is_bool_dtype(X_xgb_train[c])
    is_test_num = pd.api.types.is_numeric_dtype(X_xgb_test_full[c]) or pd.api.types.is_bool_dtype(X_xgb_test_full[c])
    if is_train_num and is_test_num:
        common_cols_numeric.append(c)
    else:
        removed_non_numeric_cols.append(c)

if excluded_precip_diag_cols:
    print("Colunas diagnósticas de acumulado excluídas do XGBoost:")
    print(excluded_precip_diag_cols)

if removed_non_numeric_cols:
    print("Colunas removidas do XGBoost por não serem numéricas:")
    print(removed_non_numeric_cols)

common_cols = common_cols_numeric
X_xgb_train = X_xgb_train[common_cols]
X_xgb_test = X_xgb_test_full[common_cols]

mask_valid_train = X_xgb_train.notna().all(axis=1) & y_xgb_train.notna()
X_xgb_train = X_xgb_train.loc[mask_valid_train]
y_xgb_train = y_xgb_train.loc[mask_valid_train]

X_xgb_test = X_xgb_test.ffill().fillna(0)

# Conversão final de segurança: todas as features do XGB devem ser numéricas.
X_xgb_train = X_xgb_train.astype(np.float32)
X_xgb_test = X_xgb_test.astype(np.float32)

print("Shape treino XGB (hard cases):", X_xgb_train.shape)
print("Shape teste XGB:", X_xgb_test.shape)

# ============================================================
# 9. VALIDAÇÃO E TREINO XGB
# ============================================================

print("\n[6/10] Separando validação temporal interna do XGBoost...")

split_idx = int(len(X_xgb_train) * 0.8)

X_train_xgb_fit = X_xgb_train.iloc[:split_idx].copy()
y_train_xgb_fit = y_xgb_train.iloc[:split_idx].copy()

X_valid_xgb = X_xgb_train.iloc[split_idx:].copy()
y_valid_xgb = y_xgb_train.iloc[split_idx:].copy()

if len(X_valid_xgb) < 30:
    raise ValueError("Conjunto de validação ficou pequeno demais após filtrar os hard cases.")

xgb_init = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=1.5,
    objective="reg:squarederror",
    random_state=42
)

xgb_init.fit(
    X_train_xgb_fit,
    y_train_xgb_fit,
    eval_set=[(X_valid_xgb, y_valid_xgb)],
    verbose=False
)

top_features = select_top_features_by_gain(
    xgb_init,
    X_train_xgb_fit,
    top_k=TOP_K_FEATURES
)

X_train_xgb_fit = X_train_xgb_fit[top_features]
X_valid_xgb = X_valid_xgb[top_features]
X_xgb_test = X_xgb_test[top_features]

print("Top features selecionadas:")
print(top_features)

print("\n[7/10] Treinando XGBoost final com early stopping...")

xgb_model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=1.5,
    objective="reg:squarederror",
    random_state=42,
    early_stopping_rounds=80
)

xgb_model.fit(
    X_train_xgb_fit,
    y_train_xgb_fit,
    eval_set=[(X_valid_xgb, y_valid_xgb)],
    verbose=False
)

# previsão residual do XGB em 2024
pred_resid_xgb_2024 = pd.Series(
    xgb_model.predict(X_xgb_test),
    index=X_xgb_test.index,
    name="pred_resid_xgb"
).reindex(y_test.index).fillna(0)

pred_resid_xgb_2024 = clip_series_by_train_quantiles(
    pred_resid_xgb_2024,
    train_ref=y_xgb_train,
    q_inf=CLIP_PREVISAO_QUANTIL_INF,
    q_sup=CLIP_PREVISAO_QUANTIL_SUP
)

# Também gera previsão do XGB em 2023 para calibrar lambda.
# A calibração pode usar todas as janelas pós-chuva, não apenas os hard cases.
if CALIBRAR_LAMBDA_EM_TODOS_EVENTOS_2023:
    idx_lambda_train = event_mask_2023[event_mask_2023].index
else:
    idx_lambda_train = X_xgb_train.index

idx_lambda_train = pd.Index(idx_lambda_train).intersection(X_xgb_train_full.index)
X_lambda_train_full = X_xgb_train_full.loc[idx_lambda_train, common_cols].copy()
X_lambda_train_full = X_lambda_train_full.ffill().fillna(0).astype(np.float32)
X_lambda_train = X_lambda_train_full[top_features]

pred_resid_xgb_train = pd.Series(
    xgb_model.predict(X_lambda_train),
    index=X_lambda_train.index,
    name="pred_resid_xgb_train"
)

pred_resid_xgb_train = clip_series_by_train_quantiles(
    pred_resid_xgb_train,
    train_ref=y_xgb_train,
    q_inf=CLIP_PREVISAO_QUANTIL_INF,
    q_sup=CLIP_PREVISAO_QUANTIL_SUP
)

# ============================================================
# 10. MATRIZ DE ESTADO x_t + LAMBDA
# ============================================================

print("\n[8/10] Estimando matriz de estado x_t e lambda por estado...")

pred_state_bins = fit_quantile_bins(
    X_xgb_train_full.loc[pred_resid_xgb_train.index, "pred_sarimax"],
    q=5,
)
state_train_df = build_state_dataframe(
    X_xgb_train_full.loc[pred_resid_xgb_train.index],
    pred_bins=pred_state_bins,
)
state_test_df = build_state_dataframe(
    X_xgb_test_full.loc[y_test.index],
    pred_bins=pred_state_bins,
)

lambda_df = estimate_lambda_by_state(
    state_df=state_train_df,
    y_true_resid=resid_oos_2023.loc[pred_resid_xgb_train.index],
    y_pred_resid=pred_resid_xgb_train,
    min_obs=LAMBDA_MIN_OBS_ESTADO,
    shrink_strength=LAMBDA_SHRINK_STRENGTH
)

diag_estado_medio_2023 = diagnostico_estado_lambda(
    state_df=state_train_df,
    y_true_resid=resid_oos_2023.loc[pred_resid_xgb_train.index],
    y_pred_resid=pred_resid_xgb_train,
    lambda_df=lambda_df,
    level="estado_medio",
)

print(f"Lambda global bruto diagnóstico, não usado como fallback dominante: {lambda_df.attrs.get('lambda_global_raw', np.nan):.4f}")
print(f"Fallback lambda neutro: {lambda_df.attrs.get('lambda_fallback', LAMBDA_PRIOR_NEUTRO):.4f}")
print("\nTop 15 estados lambda contextuais:")
if len(lambda_df):
    print(lambda_df.sort_values(["usable", "n_obs"], ascending=[False, False]).head(15))
else:
    print("lambda_df vazio.")

pred_resid_lambda_2024, lambda_t_2024 = apply_lambda_by_state(
    state_df=state_test_df,
    pred_resid_xgb=pred_resid_xgb_2024,
    lambda_df=lambda_df
)

lambda_level_2024 = lambda_t_2024.attrs.get(
    "lambda_level",
    pd.Series("fallback_1", index=lambda_t_2024.index, name="lambda_level")
)

# ============================================================
# 11. PREVISÃO HÍBRIDA FINAL
# ============================================================

print("\n[9/10] Gerando previsão híbrida final com lambda(x_t)...")

if APLICAR_CORRECAO_SOMENTE_EM_EVENTOS:
    pred_resid_xgb_2024 = pred_resid_xgb_2024.where(event_mask_2024, 0.0)
    pred_resid_lambda_2024 = pred_resid_lambda_2024.where(event_mask_2024, 0.0)

if APLICAR_LAMBDA_SOMENTE_EM_EVENTOS:
    lambda_t_2024 = lambda_t_2024.where(event_mask_2024, 1.0)

pred_hybrid_2024 = pred_sarimax_2024 + pred_resid_lambda_2024
resid_hybrid_2024 = y_test - pred_hybrid_2024

# comparação sem lambda, só para diagnóstico
pred_hybrid_sem_lambda_2024 = pred_sarimax_2024 + pred_resid_xgb_2024

# Baseline de persistência: para o alvo tau=t+1, usa somente Q_t.
pred_persistence_2024 = y.shift(1).reindex(y_test.index).astype(float)
if pred_persistence_2024.isna().any():
    raise RuntimeError("Baseline de persistência contém valores ausentes.")

# Neste run definitivo a LSTM NÃO é carregada de arquivo antigo.
# Ela será retreinada do zero após o fechamento do SARIMAX/híbrido, usando
# exatamente a mesma série y deste run. Isso elimina qualquer divergência
# de arredondamento/alinhamento entre arquivos independentes.
pred_lstm_2024 = None
lstm_source = "RETRAINED_FROM_SAME_SERIES_IN_FINAL_APPENDIX"

# ============================================================
# 12. MÉTRICAS
# ============================================================

print_metrics_table(
    title="PERFORMANCE SUMMARY - TEST PERIOD (2024 COMPLETO)",
    y_true=y_test,
    pred_sarimax=pred_sarimax_2024,
    pred_hybrid=pred_hybrid_2024,
    label_a="SARIMAX",
    label_b="HYBRID XGB + LAMBDA CONTEXTUAL"
)

print("\n" + "=" * 106)
print("COMPARAÇÃO FINAL — MESMO ALVO HORÁRIO DE 2024")
print(f"{'Modelo':<42} | {'NSE':>10} | {'RMSE':>12} | {'Spearman':>12}")
print("-" * 106)
metric_rows_print = []
for model_name, pred_model in [
    ("Persistência Q_t", pred_persistence_2024),
    ("SARIMAX rolling-state T+1", pred_sarimax_2024),
    ("Hybrid SARIMAX-XGBoost-lambda", pred_hybrid_2024),
    ("LSTM", pred_lstm_2024),
]:
    if pred_model is None:
        continue
    nse_m, rmse_m, spear_m = get_metrics(y_test, pred_model)
    metric_rows_print.append((model_name, nse_m, rmse_m, spear_m))
    print(f"{model_name:<42} | {nse_m:>10.4f} | {rmse_m:>12.4f} | {spear_m:>12.4f}")
print("=" * 106)

print_metrics_table(
    title="DIAGNÓSTICO - HYBRID SEM LAMBDA vs COM LAMBDA CONTEXTUAL",
    y_true=y_test,
    pred_sarimax=pred_hybrid_sem_lambda_2024,
    pred_hybrid=pred_hybrid_2024,
    label_a="HYBRID XGB SEM LAMBDA",
    label_b="HYBRID XGB + LAMBDA CONTEXTUAL"
)

y_test_event = y_test.loc[event_mask_2024]
pred_s_event = pred_sarimax_2024.loc[event_mask_2024]
pred_h_event = pred_hybrid_2024.loc[event_mask_2024]

if len(y_test_event) > 0:
    print_metrics_table(
        title=f"PERFORMANCE SUMMARY - JANELAS PÓS-CHUVA ({JANELA_RESPOSTA_CHUVA_HORAS}h)",
        y_true=y_test_event,
        pred_sarimax=pred_s_event,
        pred_hybrid=pred_h_event,
        label_a="SARIMAX",
        label_b="HYBRID XGB + LAMBDA CONTEXTUAL"
    )

feat_imp = pd.Series(
    xgb_model.feature_importances_,
    index=X_train_xgb_fit.columns
).sort_values(ascending=False)

print("\nTop 20 importâncias finais do XGBoost:")
print(feat_imp.head(20))

print("\nResumo do lambda_t em 2024:")
print(lambda_t_2024.describe())

# ============================================================
# 13. FIGURA INTERMEDIÁRIA DESATIVADA
# ============================================================
# Não gera uma figura provisória nesta etapa. As Figuras 7, 8 e 9 são produzidas
# somente depois do retreino da LSTM-Q e das auditorias de latência, evitando
# arquivos duplicados/desatualizados no Drive.
saida_figura = None
saida_figura_pdf = None

# ============================================================
# 14. SALVAR RESULTADOS E RESUMO FINAL
# ============================================================

series_out = pd.DataFrame(index=y_test.index)
series_out["y_real"] = y_test
series_out["pred_sarimax"] = pred_sarimax_2024
series_out["resid_sarimax"] = resid_sarimax_2024
series_out["pred_resid_xgb"] = pred_resid_xgb_2024
series_out["pred_resid_lambda"] = pred_resid_lambda_2024
series_out["lambda_t"] = lambda_t_2024
series_out["lambda_level"] = lambda_level_2024
series_out["pred_hybrid"] = pred_hybrid_2024
series_out["resid_hybrid"] = resid_hybrid_2024
series_out["pred_hybrid_sem_lambda"] = pred_hybrid_sem_lambda_2024
series_out["event_mask_14h"] = event_mask_2024.astype(int)
series_out["pred_persistence"] = pred_persistence_2024
if pred_lstm_2024 is not None:
    series_out["pred_lstm"] = pred_lstm_2024
    series_out["resid_lstm"] = y_test - pred_lstm_2024

nse_s, rmse_s, spear_s = get_metrics(y_test, pred_sarimax_2024)
nse_h, rmse_h, spear_h = get_metrics(y_test, pred_hybrid_2024)
nse_hs, rmse_hs, spear_hs = get_metrics(y_test, pred_hybrid_sem_lambda_2024)
nse_p, rmse_p, spear_p = get_metrics(y_test, pred_persistence_2024)

metrics_out = pd.DataFrame([
    {"Modelo": "PERSISTENCIA_Q_T", "NSE": nse_p, "RMSE": rmse_p, "Spearman": spear_p},
    {"Modelo": "SARIMAX", "NSE": nse_s, "RMSE": rmse_s, "Spearman": spear_s},
    {"Modelo": "HYBRID_XGB_SEM_LAMBDA", "NSE": nse_hs, "RMSE": rmse_hs, "Spearman": spear_hs},
    {"Modelo": "HYBRID_XGB_LAMBDA_CAUSAL_T1_LAG24", "NSE": nse_h, "RMSE": rmse_h, "Spearman": spear_h},
])
if pred_lstm_2024 is not None:
    nse_l, rmse_l, spear_l = get_metrics(y_test, pred_lstm_2024)
    metrics_out = pd.concat([
        metrics_out,
        pd.DataFrame([{
            "Modelo": "LSTM",
            "NSE": nse_l,
            "RMSE": rmse_l,
            "Spearman": spear_l,
        }]),
    ], ignore_index=True)

saida_excel = os.path.join(OUTPUT_DIR, NOME_EXCEL_SAIDA)
with pd.ExcelWriter(saida_excel, engine="openpyxl") as writer:
    metrics_out.to_excel(writer, sheet_name="metricas", index=False)
    series_out.reset_index(names="datetime").to_excel(writer, sheet_name="series_2024", index=False)
    lambda_df.to_excel(writer, sheet_name="lambda_por_estado", index=False)
    if "diag_estado_medio_2023" in globals() and isinstance(diag_estado_medio_2023, pd.DataFrame):
        diag_estado_medio_2023.to_excel(writer, sheet_name="diag_estado_medio_2023", index=False)
    state_cols_export = [c for c in state_test_df.columns if c.startswith("st_") or c.startswith("estado")]
    state_test_df[state_cols_export].reset_index(names="datetime").to_excel(writer, sheet_name="estados_2024", index=False)
    feat_imp.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_excel(writer, sheet_name="feature_importance", index=False)
    pd.DataFrame(rolling_meta.get("block_records", [])).to_excel(
        writer, sheet_name="rolling_blocks_2023", index=False
    )

audit_payload = {
    "status": "PASSED",
    "forecast_target": "Q_tau = Q_(t+1)",
    "forecast_issue_time": "tau-1 = t",
    "exogenous_alignment": "X_aligned[tau] = X_raw[tau-1]",
    "observed_flow_lags": f"Q_tau-1 through Q_tau-{TAMANHO_PACOTE_VAZAO_HORAS}",
    "residual_lags": f"e_tau-1 through e_tau-{TAMANHO_PACOTE_VAZAO_HORAS}",
    "rolling_signature": rolling_meta.get("signature"),
    "rolling_complete": rolling_meta.get("complete", False),
    "rolling_hours": rolling_meta.get("completed_hours"),
    "audit_2023": rolling_meta.get("audit_2023"),
    "audit_2024": audit_2024,
    "audit_xgboost_lags": "PASSED: vazao_lag_1=Q_tau-1; resid_lag_1=e_tau-1",
    "train_end": str(y_train_total.index.max()),
    "test_start": str(y_test.index.min()),
    "test_end": str(y_test.index.max()),
    "lstm_source": lstm_source,
    "generated_at": datetime.now().isoformat(),
}
saida_auditoria = OUTPUT_DIR / NOME_AUDITORIA
_atomic_write_json(audit_payload, saida_auditoria)

sse_s = float(np.sum((y_test - pred_sarimax_2024) ** 2))
sse_h = float(np.sum((y_test - pred_hybrid_2024) ** 2))
red_sse = 100.0 * (1.0 - sse_h / sse_s)

print()
print("#" * 110)
print("RESUMO FINAL - ROLLING CAUSAL T+1 COM CHECKPOINT E AUDITORIA")
print("#" * 110)
print(f"Persistência NSE:           {nse_p:.4f} | RMSE: {rmse_p:.2f} | Spearman: {spear_p:.4f}")
print(f"SARIMAX NSE:               {nse_s:.4f} | RMSE: {rmse_s:.2f} | Spearman: {spear_s:.4f}")
print(f"HYBRID sem lambda NSE:     {nse_hs:.4f} | RMSE: {rmse_hs:.2f} | Spearman: {spear_hs:.4f}")
print(f"HYBRID lambda contextual:  {nse_h:.4f} | RMSE: {rmse_h:.2f} | Spearman: {spear_h:.4f}")
if pred_lstm_2024 is not None:
    print(f"LSTM NSE:                  {nse_l:.4f} | RMSE: {rmse_l:.2f} | Spearman: {spear_l:.4f}")
print(f"Ganho NSE vs SARIMAX:      {nse_h - nse_s:.4f}")
print(f"Redução SSE vs SARIMAX:    {red_sse:.2f}%")
print(f"Auditoria causal:          APROVADA")
print(f"Checkpoint rolling:        {rolling_pred_path}")
print(f"Arquivos salvos em:        {OUTPUT_DIR}")
print(f"Excel:                     {saida_excel}")
print(f"Auditoria JSON:            {saida_auditoria}")
print("#" * 110)


# ==================================================================================================
# 15. RETREINO FINAL LSTM-Q + LATÊNCIA FIXED-MODEL + 3 FIGURAS
# ==================================================================================================
# Este bloco começa SOMENTE depois que SARIMAX final e XGBoost/lambda foram retreinados acima.
#
# PRINCÍPIOS:
#   - usa a MESMA série y de todo o pipeline;
#   - LSTM-Q é retreinada UMA ÚNICA VEZ em 2022–2023 (5 seeds);
#   - após o ajuste, LSTM, scalers, SARIMAX, XGBoost e lambda permanecem congelados;
#   - para cada latência d, somente a informação hidráulica disponível é envelhecida;
#   - hidrometeorologia disponível até t permanece inalterada;
#   - 2024 é somente avaliação nos mesmos 8.784 alvos;
#   - gera Figuras 7, 8 e 9 sem títulos descritivos internos, apenas rótulos (a), (b), (c);
#   - salva PNG/TIFF 1000 dpi e PDF vetorial.
# ==================================================================================================

import random
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import StandardScaler

LATENCIES_H = [0, 1, 2, 3, 4, 5, 6, 8, 12, 24]

# LSTM-Q final congelada pelo desenvolvimento do artigo.
LOOKBACK_LSTM = 24
LSTM_UNITS = 32
DENSE_UNITS = 16
DROPOUT_LSTM = 0.15
L2_LSTM = 1e-4
LSTM_LEARNING_RATE = 1e-3
LSTM_BATCH_SIZE = 128
LSTM_EPOCHS = 69
LSTM_SEEDS = [1001, 1002, 1003, 1004, 1005]

FINAL_DIR = Path(OUTPUT_DIR) / "FINAL_RERUN_FIXED_MODEL_WATER_RESEARCH"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
LSTM_MODEL_DIR = FINAL_DIR / "lstm_models_final"
LSTM_MODEL_DIR.mkdir(parents=True, exist_ok=True)

BITMAP_DPI = 1000
WEEKLY_ZOOM_START = "2024-02-10 00:00:00"
WEEKLY_ZOOM_END = "2024-02-17 23:00:00"

ABS_TOL_SARIMAX_D0 = 1e-6
ABS_TOL_HYBRID_D0 = 1e-5


def set_lstm_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_final_lstm_q() -> keras.Model:
    inp = keras.Input(shape=(LOOKBACK_LSTM, 1), name="historico_vazao")
    z = layers.LSTM(
        LSTM_UNITS,
        return_sequences=False,
        kernel_regularizer=regularizers.l2(L2_LSTM),
        recurrent_regularizer=regularizers.l2(L2_LSTM),
    )(inp)
    z = layers.Dropout(DROPOUT_LSTM)(z)
    z = layers.Dense(
        DENSE_UNITS,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_LSTM),
    )(z)
    out = layers.Dense(1, activation="linear")(z)
    model = keras.Model(inp, out, name="LSTM_Q_FINAL_1x32_L24")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LSTM_LEARNING_RATE),
        loss="mse",
    )
    return model


def build_lstm_samples_fixed_latency(
    full_series: pd.Series,
    target_index: pd.DatetimeIndex,
    latency_h: int,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
):
    """
    Pacote de avaliação FIXED-MODEL.

    Para o alvo Q_tau, a janela termina em Q_(tau-1-d):
      d=0  -> última vazão = Q_(tau-1) = Q_t
      d=1  -> última vazão = Q_(tau-2) = Q_(t-1)
      ...
      d=24 -> última vazão = Q_(tau-25) = Q_(t-24)

    O modelo e os scalers NÃO mudam com d.
    """
    s = pd.Series(full_series, copy=False).astype(float).sort_index()
    pos = pd.Series(np.arange(len(s), dtype=int), index=s.index)
    target_pos = pos.reindex(target_index)
    if target_pos.isna().any():
        bad = target_index[target_pos.isna()][:5]
        raise ValueError(f"Alvos LSTM fora da série completa: {list(bad)}")

    vals = s.to_numpy(dtype=float)
    X_arr = np.empty((len(target_index), LOOKBACK_LSTM, 1), dtype=np.float32)
    y_arr = np.empty(len(target_index), dtype=np.float32)

    for j, p in enumerate(target_pos.to_numpy(dtype=int)):
        end = p - int(latency_h)  # exclusive; newest = p-1-d
        start = end - LOOKBACK_LSTM
        if start < 0:
            raise ValueError(
                f"Histórico insuficiente para target={target_index[j]}, d={latency_h}."
            )
        window = vals[start:end].reshape(-1, 1)
        if len(window) != LOOKBACK_LSTM or not np.isfinite(window).all():
            raise ValueError(
                f"Janela LSTM inválida para target={target_index[j]}, d={latency_h}."
            )
        X_arr[j, :, 0] = scaler_x.transform(window).reshape(-1).astype(np.float32)
        y_arr[j] = scaler_y.transform([[vals[p]]])[0, 0]

    return X_arr, y_arr


def audit_lstm_latest_flow(
    X_arr: np.ndarray,
    target_index: pd.DatetimeIndex,
    latency_h: int,
    full_series: pd.Series,
    scaler_x: StandardScaler,
) -> float:
    """Confirma que a última vazão da janela é exatamente Q_(tau-1-d)."""
    d = int(latency_h)
    sample_ids = np.unique(np.linspace(0, len(target_index) - 1, 40, dtype=int))
    errors = []
    for j in sample_ids:
        tau = target_index[j]
        expected_time = tau - pd.Timedelta(hours=d + 1)
        qz = float(X_arr[j, -1, 0])
        q = qz * float(scaler_x.scale_[0]) + float(scaler_x.mean_[0])
        expected = float(full_series.loc[expected_time])
        errors.append(abs(q - expected))
    err = float(np.max(errors))
    if err > 1e-3:
        raise AssertionError(
            f"AUDITORIA LSTM d={d} falhou: última vazão errada; max |Δ|={err:.8f} L/s"
        )
    return err


def train_lstm_q_once_and_predict_all_latencies():
    """
    Treina as cinco redes UMA ÚNICA VEZ em d=0, usando 2022–2023.
    Em seguida, congela pesos/scalers e avalia todos os d sem novo fit.
    """
    serie = pd.Series(y, copy=False).astype(float).sort_index()
    train_start = pd.Timestamp("2022-01-01 00:00:00")
    train_end = pd.Timestamp("2023-12-31 23:00:00")
    test_start = pd.Timestamp("2024-01-01 00:00:00")
    test_end = pd.Timestamp("2024-12-31 23:00:00")

    dev_series = serie.loc[train_start:train_end]
    if dev_series.isna().any():
        raise RuntimeError("A série de desenvolvimento da LSTM contém NaN.")

    scaler_x = StandardScaler().fit(dev_series.to_numpy(dtype=float).reshape(-1, 1))
    scaler_y = StandardScaler().fit(dev_series.to_numpy(dtype=float).reshape(-1, 1))
    joblib.dump(
        {"scaler_x": scaler_x, "scaler_y": scaler_y, "lookback": LOOKBACK_LSTM},
        LSTM_MODEL_DIR / "scalers_final.joblib",
    )

    idx_dev = pd.date_range(
        train_start + pd.Timedelta(hours=LOOKBACK_LSTM),
        train_end,
        freq="h",
    )
    idx_test = pd.date_range(test_start, test_end, freq="h")

    # Treino sempre no regime normal d=0.
    X_dev, y_dev = build_lstm_samples_fixed_latency(
        serie, idx_dev, 0, scaler_x, scaler_y
    )

    # Pacotes de teste para todas as latências; pesos/scalers serão os mesmos.
    test_packages = {}
    audit_rows = []
    for d in LATENCIES_H:
        X_d, _ = build_lstm_samples_fixed_latency(
            serie, idx_test, d, scaler_x, scaler_y
        )
        err = audit_lstm_latest_flow(X_d, idx_test, d, serie, scaler_x)
        test_packages[d] = X_d
        audit_rows.append({"latency_h": d, "max_latest_flow_error_L_s": err})

    print("\n" + "#" * 100)
    print("RETREINANDO LSTM-Q FINAL UMA ÚNICA VEZ — 5 SEEDS — d=0")
    print(f"Amostras desenvolvimento: {len(X_dev)} | alvos teste: {len(idx_test)}")
    print("Depois do treino, os mesmos pesos/scalers serão usados em TODAS as latências.")
    print("#" * 100)

    pred_by_latency_seed = {d: [] for d in LATENCIES_H}

    for seed in LSTM_SEEDS:
        print(f"  seed {seed} ...")
        set_lstm_seed(seed)
        keras.backend.clear_session()
        model = build_final_lstm_q()
        model.fit(
            X_dev,
            y_dev,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            shuffle=False,
            verbose=0,
        )
        model_path = LSTM_MODEL_DIR / f"LSTM_Q_FINAL_seed{seed}.keras"
        model.save(model_path)

        for d in LATENCIES_H:
            pred_scaled = model.predict(
                test_packages[d], batch_size=1024, verbose=0
            ).reshape(-1, 1)
            pred_real = scaler_y.inverse_transform(pred_scaled).reshape(-1)
            pred_by_latency_seed[d].append(pred_real)

        del model
        keras.backend.clear_session()
        gc.collect()

    curves = {}
    for d in LATENCIES_H:
        matrix = np.column_stack(pred_by_latency_seed[d])
        out = pd.DataFrame(index=idx_test)
        out.index.name = "datetime"
        out["y_real"] = serie.reindex(idx_test).to_numpy(dtype=float)
        for j, seed in enumerate(LSTM_SEEDS):
            out[f"pred_seed_{seed}"] = matrix[:, j]
        out["pred_lstm"] = matrix.mean(axis=1)
        out["std_lstm"] = matrix.std(axis=1, ddof=1)
        out["resid_lstm"] = out["y_real"] - out["pred_lstm"]
        out["latency_h"] = d
        out.to_parquet(FINAL_DIR / f"lstm_fixed_model_curve_d{d:02d}h.parquet")
        curves[d] = out

    pd.DataFrame(audit_rows).to_csv(
        FINAL_DIR / "audit_lstm_latency_alignment.csv", index=False
    )

    # Garantia adicional: não há retreino por latência; são 5 arquivos, um por seed.
    saved_models = sorted(LSTM_MODEL_DIR.glob("LSTM_Q_FINAL_seed*.keras"))
    if len(saved_models) != len(LSTM_SEEDS):
        raise AssertionError(
            f"Esperados {len(LSTM_SEEDS)} modelos LSTM finais; encontrados {len(saved_models)}."
        )

    return curves


# --------------------------------------------------------------------------------------------------
# 15.1 RETREINAR LSTM-Q UMA VEZ E CONGELAR PARA TODAS AS LATÊNCIAS
# --------------------------------------------------------------------------------------------------
lstm_curves = train_lstm_q_once_and_predict_all_latencies()

pred_lstm_2024_final = pd.Series(
    lstm_curves[0]["pred_lstm"].to_numpy(dtype=float),
    index=y_test.index,
    name="pred_lstm",
)

nse_lstm_annual, rmse_lstm_annual, spear_lstm_annual = get_metrics(
    y_test, pred_lstm_2024_final
)
print("\nLSTM-Q anual recém-treinada:")
print(
    f"NSE={nse_lstm_annual:.6f} | RMSE={rmse_lstm_annual:.4f} | "
    f"Spearman={spear_lstm_annual:.6f}"
)

# Regime final previamente aceito; tolerância apenas para detectar arquivo/protocolo errado.
if abs(nse_lstm_annual - 0.923715) > 0.005:
    raise AssertionError(
        "A LSTM-Q recém-treinada não reproduziu o regime final esperado "
        f"(NSE ~0.923715); obtido {nse_lstm_annual:.6f}."
    )

# --------------------------------------------------------------------------------------------------
# 15.2 SARIMAX FIXED-LAG — MODELO RECÉM-TREINADO, ESTADOS FILTRADOS
# --------------------------------------------------------------------------------------------------
print("\nUsando estados do MESMO ajuste SARIMAX final — nenhum refit adicional para latência...")

filtered_states = pd.concat(
    [SARIMAX_LAT_TRAIN_STATES, SARIMAX_LAT_TEST_STATES]
).sort_index()

Z_LAT = np.asarray(SARIMAX_LAT_Z, dtype=float)
T_LAT = np.asarray(SARIMAX_LAT_T, dtype=float)
C_LAT = np.asarray(SARIMAX_LAT_C, dtype=float)
D_LAT = np.asarray(SARIMAX_LAT_D, dtype=float)

if Z_LAT.ndim == 3:
    if Z_LAT.shape[-1] != 1:
        raise RuntimeError("Design SARIMAX time-varying inesperado.")
    Z_LAT = Z_LAT[..., 0]
if T_LAT.ndim == 3:
    if T_LAT.shape[-1] != 1:
        raise RuntimeError("Transition SARIMAX time-varying inesperada.")
    T_LAT = T_LAT[..., 0]
if C_LAT.ndim == 2:
    if C_LAT.shape[-1] != 1:
        raise RuntimeError("State intercept SARIMAX time-varying inesperado.")
    C_LAT = C_LAT[..., 0]

Z_LAT = np.asarray(Z_LAT, dtype=float)
T_LAT = np.asarray(T_LAT, dtype=float)
C_LAT = np.asarray(C_LAT, dtype=float).reshape(-1)

if D_LAT.ndim == 1:
    D_TEST_LAT = np.repeat(D_LAT.reshape(-1, 1), len(y_test), axis=1)
elif D_LAT.ndim == 2 and D_LAT.shape[1] == 1:
    D_TEST_LAT = np.repeat(D_LAT, len(y_test), axis=1)
elif D_LAT.ndim == 2 and D_LAT.shape[1] == len(y_test):
    D_TEST_LAT = D_LAT
else:
    raise RuntimeError(f"Formato inesperado de obs_intercept: {D_LAT.shape}")

def propagate_sarimax_state(alpha_filtered: np.ndarray, steps: int) -> np.ndarray:
    a = np.asarray(alpha_filtered, dtype=float).reshape(-1)
    for _ in range(int(steps)):
        a = T_LAT @ a + C_LAT
    return a


def predict_sarimax_fixed_latency(d: int) -> pd.Series:
    d = int(d)
    vals = np.empty(len(y_test), dtype=float)
    for i, tau in enumerate(y_test.index):
        source_time = tau - pd.Timedelta(hours=d + 1)
        if source_time not in filtered_states.index:
            raise KeyError(
                f"Estado filtrado indisponível em {source_time}; tau={tau}, d={d}."
            )
        alpha_source = filtered_states.loc[source_time].to_numpy(dtype=float)
        alpha_tau = propagate_sarimax_state(alpha_source, d + 1)
        vals[i] = float((Z_LAT @ alpha_tau + D_TEST_LAT[:, i]).reshape(-1)[0])
    return pd.Series(vals, index=y_test.index, name=f"pred_sarimax_d{d}")


pred_sarimax_d0_check = predict_sarimax_fixed_latency(0)
max_diff_sar_d0 = float(np.max(np.abs(pred_sarimax_d0_check - pred_sarimax_2024)))
if max_diff_sar_d0 > ABS_TOL_SARIMAX_D0:
    raise AssertionError(
        f"AUDITORIA d=0 SARIMAX falhou: diferença máxima={max_diff_sar_d0:.6g}"
    )
print(f"AUDITORIA SARIMAX d=0 APROVADA: max |Δ|={max_diff_sar_d0:.3e}")


# --------------------------------------------------------------------------------------------------
# 15.3 HÍBRIDO FIXED-LAG — XGB/LAMBDA RECÉM-TREINADOS ACIMA, MEMÓRIA ATRASADA
# --------------------------------------------------------------------------------------------------
y_operacional_latency = pd.Series(y, copy=False).sort_index().astype(float)


def predict_hybrid_fixed_latency(d: int, pred_sarimax_latency: pd.Series):
    d = int(d)
    resid_sarimax_latency_2024 = y_test - pred_sarimax_latency
    resid_full_latency = pd.concat(
        [resid_oos_2023, resid_sarimax_latency_2024]
    ).sort_index()

    # add_xgb_features já cria lags 1..24. Ao deslocar a fonte por d horas,
    # vazao_lag_1 passa a representar Q_(tau-1-d), e resid_lag_1, e_(tau-1-d).
    y_source_delayed = y_operacional_latency.shift(d)
    resid_source_delayed = resid_full_latency.shift(d)

    X_base = X_test[xgb_whitelist].copy()
    X_full = add_xgb_features(
        X_base=X_base,
        pred_base=pred_sarimax_latency,
        y_real=y_source_delayed,
        resid_hist_source=resid_source_delayed,
    )

    absent_common = [c for c in common_cols if c not in X_full.columns]
    if absent_common:
        raise KeyError(f"Features ausentes em d={d}: {absent_common}")

    X_num = X_full[common_cols].copy().ffill().fillna(0).astype(np.float32)
    absent_top = [c for c in top_features if c not in X_num.columns]
    if absent_top:
        raise KeyError(f"Top features ausentes em d={d}: {absent_top}")

    pred_resid_xgb_d = pd.Series(
        xgb_model.predict(X_num[top_features]),
        index=y_test.index,
        name=f"pred_resid_xgb_d{d}",
    ).fillna(0.0)

    pred_resid_xgb_d = clip_series_by_train_quantiles(
        pred_resid_xgb_d,
        train_ref=y_xgb_train,
        q_inf=CLIP_PREVISAO_QUANTIL_INF,
        q_sup=CLIP_PREVISAO_QUANTIL_SUP,
    )

    state_df_d = build_state_dataframe(
        X_full.loc[y_test.index],
        pred_bins=pred_state_bins,
    )
    pred_resid_lambda_d, lambda_t_d = apply_lambda_by_state(
        state_df=state_df_d,
        pred_resid_xgb=pred_resid_xgb_d,
        lambda_df=lambda_df,
    )

    if APLICAR_CORRECAO_SOMENTE_EM_EVENTOS:
        pred_resid_xgb_d = pred_resid_xgb_d.where(event_mask_2024, 0.0)
        pred_resid_lambda_d = pred_resid_lambda_d.where(event_mask_2024, 0.0)
    if APLICAR_LAMBDA_SOMENTE_EM_EVENTOS:
        lambda_t_d = lambda_t_d.where(event_mask_2024, 1.0)

    pred_h = pred_sarimax_latency + pred_resid_lambda_d
    pred_h.name = f"pred_hybrid_d{d}"
    return pred_h


pred_hybrid_d0_check = predict_hybrid_fixed_latency(0, pred_sarimax_d0_check)
max_diff_hyb_d0 = float(np.max(np.abs(pred_hybrid_d0_check - pred_hybrid_2024)))
if max_diff_hyb_d0 > ABS_TOL_HYBRID_D0:
    raise AssertionError(
        f"AUDITORIA d=0 híbrido falhou: diferença máxima={max_diff_hyb_d0:.6g}"
    )
print(f"AUDITORIA HÍBRIDO d=0 APROVADA: max |Δ|={max_diff_hyb_d0:.3e}")

pred_lstm_d0_check = pd.Series(
    lstm_curves[0]["pred_lstm"].to_numpy(dtype=float),
    index=y_test.index,
)
max_diff_lstm_d0 = float(np.max(np.abs(pred_lstm_d0_check - pred_lstm_2024_final)))
if max_diff_lstm_d0 > 1e-9:
    raise AssertionError(
        f"AUDITORIA d=0 LSTM falhou: diferença máxima={max_diff_lstm_d0:.6g}"
    )
print(f"AUDITORIA LSTM-Q d=0 APROVADA: max |Δ|={max_diff_lstm_d0:.3e}")


# --------------------------------------------------------------------------------------------------
# 15.4 EXECUTAR LATÊNCIA PARA OS TRÊS MODELOS
# --------------------------------------------------------------------------------------------------
latency_rows = []
latency_predictions = pd.DataFrame(index=y_test.index)
latency_predictions.index.name = "datetime"
latency_predictions["y_real"] = y_test

for d in LATENCIES_H:
    print("\n" + "=" * 100)
    print(f"LATÊNCIA OPERACIONAL d={d} h")
    print("=" * 100)

    pred_s_d = predict_sarimax_fixed_latency(d)
    pred_h_d = predict_hybrid_fixed_latency(d, pred_s_d)
    pred_l_d = pd.Series(
        lstm_curves[d]["pred_lstm"].to_numpy(dtype=float),
        index=y_test.index,
        name=f"pred_lstm_d{d}",
    )

    latency_predictions[f"pred_sarimax_d{d}"] = pred_s_d
    latency_predictions[f"pred_hybrid_d{d}"] = pred_h_d
    latency_predictions[f"pred_lstm_d{d}"] = pred_l_d

    for model_name, pred in [
        ("SARIMAX", pred_s_d),
        ("SARIMAX-XGBoost hybrid", pred_h_d),
        ("LSTM-Q", pred_l_d),
    ]:
        nse_v, rmse_v, spear_v = get_metrics(y_test, pred)
        latency_rows.append({
            "latency_h": d,
            "model": model_name,
            "NSE": float(nse_v),
            "RMSE": float(rmse_v),
            "Spearman": float(spear_v),
        })

latency_metrics = pd.DataFrame(latency_rows).sort_values(
    ["latency_h", "model"]
).reset_index(drop=True)

print("\nRESULTADOS FINAIS — LATÊNCIA OPERACIONAL")
print(latency_metrics.to_string(index=False))

# Auditoria algébrica NSE x RMSE para todos os três modelos.
sst = float(np.sum((y_test - y_test.mean()) ** 2))
n_test_latency = len(y_test)
latency_metrics["NSE_from_RMSE"] = (
    1.0 - (latency_metrics["RMSE"] ** 2 * n_test_latency) / sst
)
latency_metrics["abs_NSE_check_error"] = (
    latency_metrics["NSE"] - latency_metrics["NSE_from_RMSE"]
).abs()
if latency_metrics["abs_NSE_check_error"].max() > 1e-8:
    raise AssertionError("Inconsistência interna NSE/RMSE no experimento de latência.")


# --------------------------------------------------------------------------------------------------
# 15.5 FIGURAS FINAIS — ARQUIVOS SEPARADOS PARA SUBMISSÃO
# --------------------------------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "mathtext.fontset": "dejavuserif",
})


def save_journal_figure(fig, stem: str):
    png = FINAL_DIR / f"{stem}.png"
    tiff = FINAL_DIR / f"{stem}.tiff"
    pdf = FINAL_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=BITMAP_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(
        tiff,
        dpi=BITMAP_DPI,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    return png, tiff, pdf


# FIGURE 7 — comparação anual SARIMAX / híbrido / LSTM
nse_s_final, rmse_s_final, spear_s_final = get_metrics(y_test, pred_sarimax_2024)
nse_h_final, rmse_h_final, spear_h_final = get_metrics(y_test, pred_hybrid_2024)

# Auditoria de identidade do regime final. Se isto falhar, NÃO use as figuras.
if nse_s_final < 0.90 or abs(nse_s_final - 0.932656) > 0.005:
    raise AssertionError(
        f"SARIMAX final inesperado: NSE={nse_s_final:.6f}; esperado ~0.932656 (>0.90)."
    )
if abs(nse_h_final - 0.929956) > 0.005:
    raise AssertionError(
        f"Híbrido final inesperado: NSE={nse_h_final:.6f}; esperado ~0.929956."
    )
print("\nAUDITORIA DO REGIME FINAL ANUAL: APROVADA")
print(f"  SARIMAX NSE={nse_s_final:.6f} (>0.90 e próximo de 0.932656)")
print(f"  HYBRID  NSE={nse_h_final:.6f} (próximo de 0.929956)")
print(f"  LSTM-Q  NSE={nse_lstm_annual:.6f} (próximo de 0.923715)")

fig7, axes = plt.subplots(3, 1, figsize=(11.2, 7.7), dpi=300, sharex=True)
panels_final = [
    ("SARIMAX base model", pred_sarimax_2024, "#ff2b2b", "SARIMAX prediction", nse_s_final),
    ("Hybrid SARIMAX-XGBoost model", pred_hybrid_2024, "#2db58c", "Hybrid prediction", nse_h_final),
    ("LSTM-Q model", pred_lstm_2024_final, "#2457ff", "LSTM-Q prediction", nse_lstm_annual),
]
all_vals = [y_test.to_numpy(dtype=float)] + [p[1].to_numpy(dtype=float) for p in panels_final]
ymin = min(float(np.nanmin(v)) for v in all_vals)
ymax = max(float(np.nanmax(v)) for v in all_vals)
ypad = 0.03 * (ymax - ymin)

for i, (ax, (label, pred, color, pred_label, nse_val)) in enumerate(zip(axes, panels_final)):
    ax.plot(y_test.index, y_test.values, color="#b8b8b8", linewidth=0.45,
            alpha=0.72, label="Measured inflow", zorder=1)
    ax.plot(pred.index, pred.values, color=color, linewidth=0.55,
            alpha=0.90, label=pred_label, zorder=2)
    ax.text(0.01, 0.97, f"({chr(97+i)})", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, fontweight="bold")
    ax.set_ylabel(r"Flow rate (L s$^{-1}$)")
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_xlim(y_test.index.min(), y_test.index.max())
    ax.grid(True, linewidth=0.45, alpha=0.30)
    ax.legend(loc="upper right", fontsize=7.5, frameon=True)

axes[-1].set_xlabel("Testing timeline (hourly data)")
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig7.tight_layout(h_pad=0.85)
fig7_paths = save_journal_figure(fig7, "Figure_7_annual_comparison_2024")
# Sobrescreve também a figura comparativa histórica com a versão correta de 3 painéis.
fig7.savefig(OUTPUT_DIR / NOME_FIGURA_SAIDA, dpi=BITMAP_DPI, bbox_inches="tight", facecolor="white")
fig7.savefig(OUTPUT_DIR / NOME_FIGURA_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig7)


# FIGURE 8 — resíduos anuais + zoom semanal
resid_lstm_final = y_test - pred_lstm_2024_final
fig8, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.2, 7.7), dpi=300, sharex=False)
ax1.plot(y_test.index, resid_sarimax_2024, color="#f08a8a", alpha=0.55,
         label="SARIMAX residuals", linewidth=0.6)
ax1.plot(y_test.index, resid_hybrid_2024, color="#2db58c", alpha=0.75,
         label="Hybrid residuals", linewidth=0.7)
ax1.plot(y_test.index, resid_lstm_final, color="#6a86ff", alpha=0.70,
         label="LSTM residuals", linewidth=0.6)
ax1.axhline(0, color="gray", linewidth=0.8)
ax1.text(0.01, 0.97, "(a)", transform=ax1.transAxes,
         ha="left", va="top", fontsize=9.5, fontweight="bold")
ax1.set_ylabel(r"Residuals (L s$^{-1}$)")
ax1.set_xlim(y_test.index.min(), y_test.index.max())
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
for lab in ax1.get_xticklabels():
    lab.set_rotation(35)
    lab.set_ha("right")
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.30)
ax1.legend(loc="upper center", ncol=3, frameon=True)

zoom = slice(WEEKLY_ZOOM_START, WEEKLY_ZOOM_END)
ax2.plot(y_test.loc[zoom].index, y_test.loc[zoom], color="black", linewidth=1.2,
         label="Measured inflow", zorder=4)
ax2.plot(pred_sarimax_2024.loc[zoom].index, pred_sarimax_2024.loc[zoom],
         color="#ff2b2b", linestyle="--", linewidth=1.0,
         label=f"SARIMAX prediction (NSE = {nse_s_final:.4f})")
ax2.plot(pred_hybrid_2024.loc[zoom].index, pred_hybrid_2024.loc[zoom],
         color="#2db58c", linewidth=1.0,
         label=f"Hybrid prediction (NSE = {nse_h_final:.4f})")
ax2.plot(pred_lstm_2024_final.loc[zoom].index, pred_lstm_2024_final.loc[zoom],
         color="#2457ff", linewidth=1.0,
         label=f"LSTM prediction (NSE = {nse_lstm_annual:.4f})")
ax2.text(0.01, 0.97, "(b)", transform=ax2.transAxes,
         ha="left", va="top", fontsize=9.5, fontweight="bold")
ax2.set_ylabel(r"Flow rate (L s$^{-1}$)")
ax2.set_xlabel("Timeline (hourly observations)")
ax2.xaxis.set_major_locator(mdates.DayLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.30)
ax2.legend(loc="upper right", frameon=True)
fig8.tight_layout(pad=1.8, h_pad=1.8)
fig8_paths = save_journal_figure(fig8, "Figure_8_residuals_and_weekly_hydrograph_2024")
plt.close(fig8)


# FIGURE 9 — latência operacional dos três modelos
fig9, (ax3, ax4) = plt.subplots(2, 1, figsize=(8.7, 8.6), dpi=300, sharex=False)
styles = [
    ("SARIMAX", "o", "#7f7f7f", "SARIMAX"),
    ("SARIMAX-XGBoost hybrid", "o", "#1f77b4", "SARIMAX-XGBoost hybrid"),
    ("LSTM-Q", "s", "#ff7f0e", "LSTM-Q"),
]
for model, marker, color, legend_label in styles:
    sub = latency_metrics.loc[latency_metrics["model"] == model].sort_values("latency_h")
    ax3.plot(sub["latency_h"], sub["NSE"], marker=marker, linewidth=1.8,
             markersize=5, color=color, label=legend_label)
    ax4.plot(sub["latency_h"], sub["RMSE"], marker=marker, linewidth=1.8,
             markersize=5, color=color, label=legend_label)

ax3.text(0.01, 0.97, "(a)", transform=ax3.transAxes,
         ha="left", va="top", fontsize=9.5, fontweight="bold")
ax3.set_ylabel("Nash-Sutcliffe efficiency (NSE)")
ax3.set_xlabel("Inflow-observation latency (h)")
ax3.set_xticks(LATENCIES_H)
ax3.set_xlim(-0.5, 24.5)
ax3.grid(True, linestyle="--", linewidth=0.6, alpha=0.30)
ax3.legend(loc="best", frameon=True)

ax4.text(0.01, 0.97, "(b)", transform=ax4.transAxes,
         ha="left", va="top", fontsize=9.5, fontweight="bold")
ax4.set_ylabel(r"Root mean square error (RMSE; L s$^{-1}$)")
ax4.set_xlabel("Inflow-observation latency (h)")
ax4.set_xticks(LATENCIES_H)
ax4.set_xlim(-0.5, 24.5)
ax4.set_ylim(bottom=0)
ax4.grid(True, linestyle="--", linewidth=0.6, alpha=0.30)
ax4.legend(loc="best", frameon=True)
fig9.tight_layout(pad=2.0, h_pad=2.0)
fig9_paths = save_journal_figure(fig9, "Figure_9_fixed_inflow_observation_latency_2024")
plt.close(fig9)


# --------------------------------------------------------------------------------------------------
# 15.6 MÉTRICAS COMPLEMENTARES / AUDITORIA R²
# --------------------------------------------------------------------------------------------------
def mae_metric(y_true, pred):
    yy = np.asarray(y_true, dtype=float)
    pp = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(yy - pp)))


def bias_metric(y_true, pred):
    yy = np.asarray(y_true, dtype=float)
    pp = np.asarray(pred, dtype=float)
    return float(np.mean(pp - yy))


def r2_standard(y_true, pred):
    yy = np.asarray(y_true, dtype=float)
    pp = np.asarray(pred, dtype=float)
    den = np.sum((yy - yy.mean()) ** 2)
    return float(1.0 - np.sum((yy - pp) ** 2) / den)


# Aqui R² convencional e NSE têm a mesma expressão SSE/SST.
for _name, _pred in [
    ("SARIMAX", pred_sarimax_2024),
    ("SARIMAX-XGBoost hybrid", pred_hybrid_2024),
    ("LSTM-Q", pred_lstm_2024_final),
]:
    _nse, _, _ = get_metrics(y_test, _pred)
    _r2 = r2_standard(y_test, _pred)
    if abs(_nse - _r2) > 1e-12:
        raise AssertionError(f"R²/NSE inconsistente para {_name}.")


# --------------------------------------------------------------------------------------------------
# 15.7 EXPORTAR UM ÚNICO ARQUIVO DE AUDITORIA FINAL
# --------------------------------------------------------------------------------------------------
annual_series_final = pd.DataFrame(index=y_test.index)
annual_series_final.index.name = "datetime"
annual_series_final["y_real"] = y_test
annual_series_final["pred_sarimax"] = pred_sarimax_2024
annual_series_final["pred_hybrid"] = pred_hybrid_2024
annual_series_final["pred_lstm"] = pred_lstm_2024_final
annual_series_final["resid_sarimax"] = resid_sarimax_2024
annual_series_final["resid_hybrid"] = resid_hybrid_2024
annual_series_final["resid_lstm"] = resid_lstm_final

annual_metrics_final = pd.DataFrame([
    {"model": "SARIMAX", "NSE": nse_s_final, "RMSE": rmse_s_final, "Spearman": spear_s_final,
     "MAE": mae_metric(y_test, pred_sarimax_2024), "Bias": bias_metric(y_test, pred_sarimax_2024),
     "R2_standard": r2_standard(y_test, pred_sarimax_2024)},
    {"model": "SARIMAX-XGBoost hybrid", "NSE": nse_h_final, "RMSE": rmse_h_final, "Spearman": spear_h_final,
     "MAE": mae_metric(y_test, pred_hybrid_2024), "Bias": bias_metric(y_test, pred_hybrid_2024),
     "R2_standard": r2_standard(y_test, pred_hybrid_2024)},
    {"model": "LSTM-Q", "NSE": nse_lstm_annual, "RMSE": rmse_lstm_annual, "Spearman": spear_lstm_annual,
     "MAE": mae_metric(y_test, pred_lstm_2024_final), "Bias": bias_metric(y_test, pred_lstm_2024_final),
     "R2_standard": r2_standard(y_test, pred_lstm_2024_final)},
])

final_excel = FINAL_DIR / "FINAL_models_latency_and_figures_2024.xlsx"
with pd.ExcelWriter(final_excel, engine="openpyxl") as writer:
    annual_metrics_final.to_excel(writer, sheet_name="annual_metrics", index=False)
    annual_series_final.reset_index().to_excel(writer, sheet_name="annual_series", index=False)
    latency_metrics.to_excel(writer, sheet_name="latency_metrics", index=False)
    latency_predictions.reset_index().to_excel(writer, sheet_name="latency_predictions", index=False)
    feat_imp.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_excel(
        writer, sheet_name="xgb_feature_importance", index=False
    )
    lambda_df.to_excel(writer, sheet_name="lambda_by_state", index=False)
    pd.DataFrame({
        "item": [
            "protocol", "development", "test", "latencies_h",
            "lstm_lookback", "lstm_units", "lstm_dense", "lstm_dropout",
            "lstm_l2", "lstm_lr", "lstm_batch", "lstm_epochs", "lstm_seeds",
            "sarimax_order", "sarimax_seasonal_order",
            "rolling_force_rebuild", "rolling_checkpoint_used", "figure_bitmap_dpi",
            "latency_design", "lstm_retrained_per_latency",
            "sarimax_d0_max_abs_diff", "hybrid_d0_max_abs_diff", "lstm_d0_max_abs_diff",
        ],
        "value": [
            "Q_tau forecast using information available no later than tau-1; fixed inflow-observation latency d",
            "2022-01-01 through 2023-12-31",
            "2024-01-01 through 2024-12-31",
            str(LATENCIES_H),
            LOOKBACK_LSTM, LSTM_UNITS, DENSE_UNITS, DROPOUT_LSTM,
            L2_LSTM, LSTM_LEARNING_RATE, LSTM_BATCH_SIZE, LSTM_EPOCHS, str(LSTM_SEEDS),
            str(SARIMAX_ORDER), str(SARIMAX_SEASONAL_ORDER),
            FORCE_REBUILD_ROLLING, str(EXISTING_ROLLING_PARQUET), BITMAP_DPI,
            "fixed-model: fitted models/scalers/lambda frozen; only hydraulic information delayed", False,
            max_diff_sar_d0, max_diff_hyb_d0, max_diff_lstm_d0,
        ],
    }).to_excel(writer, sheet_name="configuration", index=False)

# Tabela principal pronta para o manuscrito (R² fica apenas na auditoria, pois é igual ao NSE).
table4_final = annual_metrics_final[["model", "NSE", "RMSE", "Spearman"]].copy()
table4_final.to_csv(FINAL_DIR / "Table_4_final_metrics.csv", index=False)

captions = (
    "Figure 7. Comparison of measured and one-hour-ahead predicted wastewater inflow during the 2024 "
    "chronological evaluation: (a) standalone SARIMAX; (b) SARIMAX-XGBoost hybrid; and (c) LSTM-Q.\n\n"
    "Figure 8. Residual and event-scale diagnostics during the 2024 evaluation period: (a) annual residual "
    "time series for the standalone SARIMAX, SARIMAX-XGBoost hybrid, and LSTM-Q models; and (b) observed "
    "and predicted hourly inflow during a representative weekly interval (10-17 February 2024).\n\n"
    "Figure 9. Sensitivity of one-hour-ahead wastewater inflow forecasting to increasing inflow-observation "
    "latency during the 2024 test period: (a) Nash-Sutcliffe efficiency (NSE); and (b) root mean square error (RMSE)."
)
(FINAL_DIR / "recommended_captions.txt").write_text(captions, encoding="utf-8")

# Auditoria simples para confirmar que somente 5 fits LSTM ocorreram no desenho metodológico.
run_design = pd.DataFrame({
    "item": [
        "rolling_2023_recomputed",
        "rolling_checkpoint",
        "sarimax_final_retrained",
        "xgboost_lambda_retrained",
        "lstm_final_models_trained",
        "lstm_retrained_per_latency",
        "latency_models_frozen",
        "latency_points_h",
        "hydrometeorological_inputs_delayed",
        "same_2024_targets_all_latencies",
    ],
    "value": [
        False,
        str(EXISTING_ROLLING_PARQUET),
        True,
        True,
        len(LSTM_SEEDS),
        False,
        True,
        str(LATENCIES_H),
        False,
        True,
    ],
})
run_design.to_csv(FINAL_DIR / "run_design_audit.csv", index=False)

print("\n" + "#" * 110)
print("RUN DEFINITIVO CONCLUÍDO — TODOS OS MODELOS/FIGURAS")
print("#" * 110)
print("\nMétricas anuais:")
print(annual_metrics_final.to_string(index=False))
print("\nLatência operacional:")
print(latency_metrics[["latency_h", "model", "NSE", "RMSE", "Spearman"]].to_string(index=False))
print(f"\nExcel final: {final_excel}")
print("\nFigure 7:")
for p in fig7_paths: print("  ", p)
print("\nFigure 8:")
for p in fig8_paths: print("  ", p)
print("\nFigure 9:")
for p in fig9_paths: print("  ", p)
print("#" * 110)
