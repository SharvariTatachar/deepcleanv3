import pandas as pd

def norm(ch):
    return (
        ch.strip()
        .replace(":", "_")
        .replace("-", "_")
        .upper()
    )

df = pd.read_csv("data/coherence_outputs/coherence_summary.csv")

wanted = [
    "H1:TCS_ITMY_CO2_ISS_OUT_AC_OUT_DQ",
    "H1:PEM_CS_ACC_PSL_PERISCOPE_X_DQ",
    "H1:SUS_PR3_M3_OPLEV_YAW_OUT_DQ",
    "H1:PEM_CS_RADIO_EBAY_NARROWBAND_2_DQ",
    "H1:ASC_PRC1_Y_OUT_DQ",
    "H1:ISI_HAM3_BLND_GS13X_IN1_DQ",
    "H1:PSL_ILS_HV_MON_OUT_DQ",
    "H1:PEM_FCES_ACC_HAM8_FC2_Z_DQ",
    "H1:PEM_CS_MAG_LVEA_VERTEX_X_DQ",
    "H1:SUS_SR2_M2_WIT_P_DQ",
    "H1:SUS_ZM3_M1_DAMP_L_IN1_DQ",
    "H1:ISI_ITMY_ST1_BLND_RX_T240_CUR_IN1_DQ",
    "H1:PEM_EY_SEIS_VEA_FLOOR_X_DQ",
    "H1:PEM_CS_ACC_BSC2_BS_Z_DQ",
    "H1:SUS_BS_M3_OPLEV_SUM_OUT_DQ",
    "H1:ISI_ETMX_ST1_BLND_RZ_T240_CUR_IN1_DQ",
    "H1:PEM_VAULT_SEIS_1030X195Y_STS2_Z_DQ",
    "H1:HPI_HAM5_BLND_L4C_Y_IN1_DQ",
    "H1:ISI_HAM7_BLND_GS13RZ_IN1_DQ",
    "H1:HPI_HAM5_BLND_L4C_Z_IN1_DQ",
    "H1:SUS_PR3_M2_WIT_Y_DQ",
    "H1:HPI_ETMY_BLND_L4C_RX_IN1_DQ",
    "H1:SUS_FC1_M1_DAMP_V_IN1_DQ",
    "H1:SUS_SR3_M3_OPLEV_YAW_OUT_DQ",
    "H1:ISI_HAM2_BLND_GS13RX_IN1_DQ",
    "H1:SQZ_CLF_SERVO_EXC_OUT_DQ",
    "H1:SUS_MC3_M1_DAMP_T_IN1_DQ",
    "H1:SUS_SRM_M3_WIT_P_DQ",
    "H1:SUS_ETMY_L1_WIT_Y_DQ",
    "H1:HPI_HAM6_BLND_L4C_Z_IN1_DQ",
    "H1:ISI_ITMY_ST2_BLND_RY_GS13_CUR_IN1_DQ",
    "H1:HPI_ETMY_BLND_L4C_X_IN1_DQ",
    "H1:SUS_OMC_M1_DAMP_V_IN1_DQ",
    "H1:ASC_Y_TR_A_PIT_OUT_DQ",
    "H1:SUS_MC1_M1_DAMP_Y_IN1_DQ"
]


df["channel_norm"] = df["channel"].map(norm)
wanted_norm = [norm(ch) for ch in wanted]

matches = df[df["channel_norm"].isin(wanted_norm)]

print(
    matches[["channel", "band_mean_coherence"]]
    .sort_values("band_mean_coherence", ascending=False)
)

missing = sorted(set(wanted_norm) - set(matches["channel_norm"]))
print("\nMissing:")
for ch in missing:
    print(ch)