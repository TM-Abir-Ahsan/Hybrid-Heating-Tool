import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

st.set_page_config(page_title="Hybrid Heating Analysis", layout="wide")

SCENARIO_NAMES = [
    'BAL_PCV_1800', 'BAL_PCV_2400', 'BAL_PCV_3000',
    'BAL_IECC_1800', 'BAL_IECC_2400', 'BAL_IECC_3000',
    'MAL_PCV_1800',  'MAL_PCV_2400',  'MAL_PCV_3000',
    'MAL_IECC_1800', 'MAL_IECC_2400', 'MAL_IECC_3000',
]
BAL_SCENARIOS = [s for s in SCENARIO_NAMES if 'BAL' in s]

HP_SIZES_MBH  = [s * 12 for s in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7.5, 10, 12.5, 15, 20]]
FURNACE_SIZES = [20, 25, 40, 60, 80, 100, 120, 140]
HP_REF_CAP    = 60

HYBRID_OPTIONS = [
    'NC Furnace + SS HP',        'NC Furnace + Two-stage HP',   'NC Furnace + Modulating HP',
    'Cond Furnace + SS HP',      'Cond Furnace + Two-stage HP', 'Cond Furnace + Modulating HP',
]
NON_HYBRID_OPTIONS   = ['NC Furnace', 'Cond Furnace', 'SS HP', 'Two-stage HP', 'Modulating HP']
CONTROL_STRATEGIES   = ['Capacity Switchover', 'Cost Switchover', '35F Switchover', '40F Switchover']
HYBRID_HP_STRATEGIES = ['Sized for Cooling', 'Sized for Heating']
NH_HP_STRATEGY       = 'Sized for Heating (with 1.25x cooling oversize limit)'


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading equipment & load data…")
def load_all_data(load_bytes, twostage_bytes, carrier_bytes):
    load_dfs = {s: pd.read_excel(io.BytesIO(load_bytes), sheet_name=s) for s in SCENARIO_NAMES}

    eq = {
        'SS_HP_H_Cap': pd.read_excel(io.BytesIO(twostage_bytes), sheet_name='One_stg_H_MBH'),
        'SS_HP_H_COP': pd.read_excel(io.BytesIO(twostage_bytes), sheet_name='One_stg_H_COP'),
        'SS_HP_C_Cap': pd.read_excel(io.BytesIO(twostage_bytes), sheet_name='One_stg_C_MBH'),
        'SS_HP_C_COP': pd.read_excel(io.BytesIO(twostage_bytes), sheet_name='One_stg_C_COP'),
        'TS_HP_H_Cap': pd.read_excel(io.BytesIO(carrier_bytes),  sheet_name='Two_stg_H_MBH'),
        'TS_HP_H_COP': pd.read_excel(io.BytesIO(carrier_bytes),  sheet_name='Two_stg_H_COP'),
        'VS_HP_H_Cap': pd.read_excel(io.BytesIO(carrier_bytes),  sheet_name='VSHP_H_MBH'),
        'VS_HP_H_COP': pd.read_excel(io.BytesIO(carrier_bytes),  sheet_name='VSHP_H_COP'),
    }
    return load_dfs, eq


# ── Equipment performance helpers ─────────────────────────────────────────────
def _derate(PL):
    return np.where(PL < 0.5, 1.81*PL**3 - 3.59*PL**2 + 2.37*PL + 0.406,
           np.where(PL < 1.0, 0.0844*PL + 0.8732, 1.0))

def sim_furnace(fur_cap, fur_eff, load):
    PL   = load / fur_cap
    eff  = (-0.07713*PL**2 + 0.23495*PL + 0.7922) * fur_eff / 0.95
    Gas  = load / eff
    Elec = (-0.00002*load**2 + 0.007*load + 0.011) * fur_cap / 80
    return Gas, Elec, np.zeros_like(load)

def sim_SS_HP(cap, ref, temp, load, eq):
    op_c = np.interp(temp, eq['SS_HP_H_Cap']['OAT'], eq['SS_HP_H_Cap']['Cap'])
    op_p = np.interp(temp, eq['SS_HP_H_COP']['OAT'], eq['SS_HP_H_COP']['COP'])
    sc   = op_c * (cap / ref)
    exc  = load > sc
    PL   = np.where(exc, 1.0, np.where(load > 0, load/sc, 0.0))
    sup  = np.where(exc, load - sc, 0.0)
    act  = np.where(exc, sc, load)
    E    = np.where(load > 0, ((act / (op_p * _derate(PL))) + sup) * 0.293, 0.0)
    return np.zeros_like(load, dtype=float), E, np.where(load > 0, sup, 0.0)*0.293, sc

def sim_TS_HP(cap, ref, temp, load, eq):
    cH = np.interp(temp, eq['TS_HP_H_Cap']['OAT'], eq['TS_HP_H_Cap']['High'])
    pH = np.interp(temp, eq['TS_HP_H_COP']['OAT'], eq['TS_HP_H_COP']['High'])
    cL = np.interp(temp, eq['TS_HP_H_Cap']['OAT'], eq['TS_HP_H_Cap']['Low'])
    pL = np.interp(temp, eq['TS_HP_H_COP']['OAT'], eq['TS_HP_H_COP']['Low'])
    scH, scL = cH*(cap/ref), cL*(cap/ref)
    abv = load > scH;  btw = (load >= scL) & (load <= scH);  blw = (load > 0) & (load < scL)
    sup = load - scH
    Eab = ((scH/pH) + sup) * 0.293
    sf  = (load-scL)/(scH-scL)
    Ebt = (scH/pH*sf + scL/pL*(1-sf)) * 0.293
    PLl = np.where(scL > 0, load/scL, 0.0)
    Ebl = (load / (pL * _derate(PLl))) * 0.293
    E   = np.where(abv, Eab, np.where(btw, Ebt, np.where(blw, Ebl, 0.0)))
    return np.zeros_like(load, dtype=float), E, np.where(abv, sup, 0.0)*0.293, scH

def sim_VS_HP(cap, ref, temp, load, eq):
    cX = np.interp(temp, eq['VS_HP_H_Cap']['OAT'], eq['VS_HP_H_Cap']['Max'])
    pX = np.interp(temp, eq['VS_HP_H_COP']['OAT'], eq['VS_HP_H_COP']['Max'])
    cN = np.interp(temp, eq['VS_HP_H_Cap']['OAT'], eq['VS_HP_H_Cap']['Min'])
    pN = np.interp(temp, eq['VS_HP_H_COP']['OAT'], eq['VS_HP_H_COP']['Min'])
    scX, scN = cX*(cap/ref), cN*(cap/ref)
    abv = load > scX;  btw = (load >= scN) & (load <= scX);  blw = (load > 0) & (load < scN)
    sup = load - scX
    Eab = ((scX/pX) + sup) * 0.293
    pi  = pN + (pX-pN)*((load-scN)/(scX-scN))
    Ebt = (load / pi) * 0.293
    PLn = np.where(scN > 0, load/scN, 0.0)
    Ebl = (load / (pN * _derate(PLn))) * 0.293
    E   = np.where(abv, Eab, np.where(btw, Ebt, np.where(blw, Ebl, 0.0)))
    return np.zeros_like(load, dtype=float), E, np.where(abv, sup, 0.0)*0.293, scX

def _hp_type_from_cfg(cfg):
    if 'SS HP'        in cfg: return 'Single-stage HP'
    if 'Two-stage HP' in cfg: return 'Two-stage HP'
    return 'Modulating HP'

def _run_hp(hp_type, cap, temp, load, eq):
    if hp_type == 'Single-stage HP': return sim_SS_HP(cap, HP_REF_CAP, temp, load, eq)
    if hp_type == 'Two-stage HP':    return sim_TS_HP(cap, HP_REF_CAP, temp, load, eq)
    return sim_VS_HP(cap, HP_REF_CAP, temp, load, eq)


# ── Sizing ────────────────────────────────────────────────────────────────────
def compute_sizing(load_dfs, eq):
    rows = []
    for sc in SCENARIO_NAMES:
        h_oat, c_oat = (21.6, 95.3) if 'BAL' in sc else (31.4, 96.0)
        LD  = load_dfs[sc].sort_values('OAT F')
        hi, ci = int(h_oat), int(c_oat)
        dh = LD[(LD['OAT F'] >= hi)  & (LD['OAT F'] < hi+1)]['Heating MBH'].max()
        dc = LD[(LD['OAT F'] >= ci) & (LD['OAT F'] < ci+1)]['Cooling MBH'].max()

        cC = np.interp(c_oat, eq['SS_HP_C_Cap']['OAT'], eq['SS_HP_C_Cap']['Cap'])
        cH = np.interp(h_oat, eq['SS_HP_H_Cap']['OAT'], eq['SS_HP_H_Cap']['Cap'])

        sel_cl = next((s for s in HP_SIZES_MBH if cC*(s/HP_REF_CAP) >= dc), HP_SIZES_MBH[-1])
        max_l  = sel_cl * 1.25
        sel_hc = sel_cl;  hmet_c = False
        for s in HP_SIZES_MBH:
            if s < sel_cl or s > max_l: continue
            sel_hc = s
            if cH*(s/HP_REF_CAP) >= dh: hmet_c = True; break
        sel_hu = next((s for s in HP_SIZES_MBH if cH*(s/HP_REF_CAP) >= dh), HP_SIZES_MBH[-1])
        sel_co = next((s for s in HP_SIZES_MBH if cC*(s/HP_REF_CAP) >= dc*1.15), HP_SIZES_MBH[-1])

        fl    = dh * 1.4
        sel_f = FURNACE_SIZES[0]
        for i in range(len(FURNACE_SIZES)-1):
            if fl > FURNACE_SIZES[i] and fl <= FURNACE_SIZES[i+1]:
                sel_f = FURNACE_SIZES[i+1]; break
        if fl > FURNACE_SIZES[-1]: sel_f = FURNACE_SIZES[-1]

        rows.append({
            'Scenario': sc,
            'Cooling_Load_MBH': dc, 'Heating_Load_MBH': dh,
            'Cooling_Based_Capacity_MBH': sel_cl,
            'Final_Constrained_Capacity_MBH': sel_hc,
            'Final_Constrained_Capacity_Tons': sel_hc/12,
            'Heating_Met_Constrained': hmet_c,
            'Heating_Only_Capacity_MBH': sel_hu,
            'Furnace_Selected_MBH': sel_f,
            'Cooling_Oversize_Selected_MBH': sel_co,
            'Cooling_Oversize_Selected_Tons': sel_co/12,
        })

    df = pd.DataFrame(rows).set_index('Scenario')
    df.loc['BAL_IECC_1800', 'Final_Constrained_Capacity_MBH']  = 24
    df.loc['BAL_IECC_1800', 'Final_Constrained_Capacity_Tons'] = 2.0
    return df


# ── Simulation ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running simulation…")
def run_simulation(elec_price, gas_rate, nc_eff, cond_eff, load_bytes, twostage_bytes, carrier_bytes):
    load_dfs, eq = load_all_data(load_bytes, twostage_bytes, carrier_bytes)
    sizing_df    = compute_sizing(load_dfs, eq)

    def _hp_cap(sc, strat):
        if strat == 'Sized for Cooling':    return sizing_df.loc[sc, 'Cooling_Oversize_Selected_MBH']
        if strat == NH_HP_STRATEGY:         return sizing_df.loc[sc, 'Final_Constrained_Capacity_MBH']
        return sizing_df.loc[sc, 'Heating_Only_Capacity_MBH']

    results = {};  non_hybrid_results = {};  hourly_results = {}

    for sc in BAL_SCENARIOS:
        LD   = load_dfs[sc]
        heat = np.array(LD['Heating MBH']);  oat = np.array(LD['OAT F'])
        fur_c = sizing_df.loc[sc, 'Furnace_Selected_MBH']

        # Non-hybrid
        non_hybrid_results[sc] = {NH_HP_STRATEGY: {}}
        for sys in NON_HYBRID_OPTIONS:
            hc = _hp_cap(sc, NH_HP_STRATEGY)
            if   sys == 'NC Furnace':    G, E, _ = sim_furnace(fur_c, nc_eff,   heat)
            elif sys == 'Cond Furnace':  G, E, _ = sim_furnace(fur_c, cond_eff, heat)
            elif sys == 'SS HP':         G, E, _, _ = sim_SS_HP(hc, HP_REF_CAP, oat, heat, eq)
            elif sys == 'Two-stage HP':  G, E, _, _ = sim_TS_HP(hc, HP_REF_CAP, oat, heat, eq)
            else:                        G, E, _, _ = sim_VS_HP(hc, HP_REF_CAP, oat, heat, eq)
            df_h = pd.DataFrame({'Hour': np.arange(len(heat)), 'OAT_F': oat,
                                  'Load_MBH': heat, 'Elec_kWh': E, 'Gas_MBH': G})
            tg = G.sum();  te = E.sum();  gc = tg*gas_rate;  ec = te*elec_price
            non_hybrid_results[sc][NH_HP_STRATEGY][sys] = {
                'Gas_MBH': tg, 'Electricity_kWh': te, 'Cost': gc+ec,
                'Gas_Cost': gc, 'Electricity_Cost': ec
            }
            hourly_results[f"{sc}__{NH_HP_STRATEGY}__{sys}".replace(' ','_')] = df_h

        # Hybrid
        results[sc] = {}
        for hp_strat in HYBRID_HP_STRATEGIES:
            results[sc][hp_strat] = {}
            hc  = _hp_cap(sc, hp_strat)
            f_e = nc_eff  # furnace eff determined per config below

            for cfg in HYBRID_OPTIONS:
                results[sc][hp_strat][cfg] = {}
                fur_eff  = nc_eff if cfg.startswith('NC') else cond_eff
                ht       = _hp_type_from_cfg(cfg)
                _, HP_E, _, HP_Cap = _run_hp(ht, hc, oat, heat, eq)
                Fur_G, Fur_E, _    = sim_furnace(fur_c, fur_eff, heat)

                for ctrl in CONTROL_STRATEGIES:
                    hp_can = HP_Cap >= heat
                    if ctrl == 'Capacity Switchover':
                        run_hp = hp_can
                    elif ctrl == 'Cost Switchover':
                        run_hp = hp_can & (HP_E*elec_price <= Fur_G*gas_rate + Fur_E*elec_price)
                    else:
                        thr    = float(ctrl.split('F')[0])
                        run_hp = hp_can & (oat >= thr)
                    run_f = ~run_hp

                    df_h = pd.DataFrame({
                        'Hour': np.arange(len(heat)), 'OAT_F': oat, 'Load_MBH': heat,
                        'HP_Capacity_MBH':  HP_Cap,
                        'HP_Elec_kWh':      np.where(run_hp, HP_E,  0.0),
                        'HP_Gas_MBH':       0.0,
                        'Furnace_Elec_kWh': np.where(run_f,  Fur_E, 0.0),
                        'Furnace_Gas_MBH':  np.where(run_f,  Fur_G, 0.0),
                        'Equipment_Ran':    np.where(run_hp, 'HP', 'Furnace'),
                    })
                    tg = df_h['Furnace_Gas_MBH'].sum()
                    te = df_h['HP_Elec_kWh'].sum() + df_h['Furnace_Elec_kWh'].sum()
                    gc = tg*gas_rate;  ec = te*elec_price

                    if ctrl == 'Capacity Switchover':
                        fur_h  = df_h.loc[(df_h['Equipment_Ran']=='Furnace') & (df_h['Load_MBH']>0), 'OAT_F']
                        sw_oat = float(fur_h.max()) if not fur_h.empty else None
                    elif ctrl.endswith('F Switchover'):
                        sw_oat = float(ctrl.split('F')[0])
                    else:
                        sw_oat = None

                    results[sc][hp_strat][cfg][ctrl] = {
                        'Gas_MBH': tg, 'Electricity_kWh': te, 'Cost': gc+ec,
                        'Gas_Cost': gc, 'Electricity_Cost': ec, 'Switchover_OAT_F': sw_oat
                    }
                    key = (f"{sc}__{hp_strat}__{cfg}__{ctrl}"
                           .replace(' ','_').replace('+','and'))
                    hourly_results[key] = df_h

    return results, non_hybrid_results, hourly_results, sizing_df


# ── Plot helpers ──────────────────────────────────────────────────────────────
_RC = {'font.size': 12, 'font.family': 'DejaVu Sans'}


def show_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)
_REGION_COLORS = ['lightyellow', 'lightgoldenrodyellow', 'lavender', 'honeydew', 'mistyrose']


def _add_dividers(ax, edges, ctrl_list, y_top):
    for i, ctrl in enumerate(ctrl_list):
        ax.axvspan(edges[i], edges[i+1], color=_REGION_COLORS[i % len(_REGION_COLORS)], alpha=0.45, zorder=0)
        ax.text((edges[i]+edges[i+1])/2, y_top*1.18, ctrl,
                ha='center', va='bottom', fontsize=9, color='dimgray', style='italic')
    for d in edges:
        ax.axvline(x=d, color='black', linestyle='--', linewidth=1, alpha=0.5)


def _build_cost_energy_data(sc, ctrl_list, hp_strat, results, non_hybrid_results, mode):
    """mode='cost' or 'energy'"""
    nh_items = list(non_hybrid_results[sc][NH_HP_STRATEGY].items())
    gas_sys  = nh_items[:2];  elec_sys = nh_items[-3:]
    labels = []; v1 = []; v2 = []

    for sys, r in gas_sys:
        labels.append(sys)
        if mode == 'cost':  v1.append(r['Gas_Cost']);  v2.append(r['Electricity_Cost'])
        else:               v1.append(r['Gas_MBH']);   v2.append(r['Electricity_kWh']*3.412)

    n_gas = len(gas_sys);  n_per = []
    for ctrl in ctrl_list:
        cnt = 0
        for cfg in results[sc][hp_strat]:
            r = results[sc][hp_strat][cfg][ctrl]
            labels.append(cfg)
            if mode == 'cost': v1.append(r['Gas_Cost']); v2.append(r['Electricity_Cost'])
            else:              v1.append(r['Gas_MBH']);  v2.append(r['Electricity_kWh']*3.412)
            cnt += 1
        n_per.append(cnt)

    for sys, r in elec_sys:
        labels.append(sys)
        if mode == 'cost': v1.append(r['Gas_Cost']); v2.append(r['Electricity_Cost'])
        else:              v1.append(r['Gas_MBH']);  v2.append(r['Electricity_kWh']*3.412)

    edges = [n_gas - 0.5];  cum = n_gas
    for cnt in n_per:
        cum += cnt; edges.append(cum - 0.5)
    return labels, np.array(v1), np.array(v2), edges


# ── Plot: Annual Cost ─────────────────────────────────────────────────────────
def plot_annual_cost(sc, ctrl_list, results, non_hybrid_results):
    plt.rcParams.update(_RC)
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    fig.subplots_adjust(hspace=0.25)

    for ax, hp_strat in zip(axes, HYBRID_HP_STRATEGIES):
        labels, gas_c, elec_c, edges = _build_cost_energy_data(
            sc, ctrl_list, hp_strat, results, non_hybrid_results, 'cost')
        total = gas_c + elec_c;  x = np.arange(len(labels))

        ax.bar(x, gas_c,  width=0.6, color='tomato',    edgecolor='white')
        ax.bar(x, elec_c, width=0.6, color='steelblue', edgecolor='white', bottom=gas_c)
        for i, t in enumerate(total):
            ax.text(x[i], t + total.max()*0.01, f'${t:,.0f}',
                    ha='center', va='bottom', rotation=90, fontsize=9)
        _add_dividers(ax, edges, ctrl_list, total.max())
        ax.set_xticks(x)
        if ax == axes[-1]:
            ax.set_xticklabels([l.replace(' ','\n',1) for l in labels], rotation=45, ha='center')
        else:
            ax.set_xticklabels([])
        ax.set_ylabel('Annual Cost ($)');  ax.set_ylim(0, total.max()*1.25)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7);  ax.set_axisbelow(True)
        ax.set_title(f'HP Sizing: {hp_strat}', fontsize=11, fontweight='bold', pad=8)
        ax.legend(handles=[mpatches.Patch(facecolor='tomato',    label='Gas Cost'),
                            mpatches.Patch(facecolor='steelblue', label='Elec Cost')], loc='upper right')

    fig.suptitle(f'Annual Heating Cost — {sc}', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


# ── Plot: Annual Energy ───────────────────────────────────────────────────────
def plot_annual_energy(sc, ctrl_list, results, non_hybrid_results):
    plt.rcParams.update(_RC)
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    fig.subplots_adjust(hspace=0.25)

    for ax, hp_strat in zip(axes, HYBRID_HP_STRATEGIES):
        labels, gas_k, elec_k, edges = _build_cost_energy_data(
            sc, ctrl_list, hp_strat, results, non_hybrid_results, 'energy')
        total = gas_k + elec_k;  x = np.arange(len(labels))

        ax.bar(x, gas_k,  width=0.6, color='tomato',    edgecolor='white')
        ax.bar(x, elec_k, width=0.6, color='steelblue', edgecolor='white', bottom=gas_k)
        for i, t in enumerate(total):
            ax.text(x[i], t + total.max()*0.01, f'{t:,.0f}',
                    ha='center', va='bottom', rotation=90, fontsize=9)
        for i in range(len(labels)):
            if gas_k[i]  > total.max()*0.05:
                ax.text(x[i], gas_k[i]/2,  f"{gas_k[i]/1000:.1f}\nMBtu",
                        ha='center', va='center', fontsize=7, color='white')
            if elec_k[i] > total.max()*0.05:
                ax.text(x[i], gas_k[i]+elec_k[i]/2, f"{elec_k[i]/3.412:,.0f}\nkWh",
                        ha='center', va='center', fontsize=7, color='white')
        _add_dividers(ax, edges, ctrl_list, total.max())
        ax.set_xticks(x)
        if ax == axes[-1]:
            ax.set_xticklabels([l.replace(' ','\n',1) for l in labels], rotation=45, ha='center')
        else:
            ax.set_xticklabels([])
        ax.set_ylabel('Annual Energy (kBtu)');  ax.set_ylim(0, total.max()*1.25)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7);  ax.set_axisbelow(True)
        ax.set_title(f'HP Sizing: {hp_strat}', fontsize=11, fontweight='bold', pad=8)
        ax.legend(handles=[mpatches.Patch(facecolor='tomato',    label='Gas'),
                            mpatches.Patch(facecolor='steelblue', label='Electricity')], loc='upper right')

    fig.suptitle(f'Annual Energy Demand — {sc}', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


# ── Plot: Peak Demand ─────────────────────────────────────────────────────────
def plot_peak_demand(home_size, results, hourly_results, sizing_df, eq, load_dfs):
    hp_strat    = 'Sized for Cooling'
    sc_old      = f'BAL_PCV_{home_size}'
    sc_new      = f'BAL_IECC_{home_size}'
    sc_labels   = {sc_old: f'Older Homes ({home_size} sqft)', sc_new: f'Newer Homes ({home_size} sqft)'}
    ctrl_colors = ['#4472C4', '#ED7D31', '#70AD47']
    ctrl_labels = ['35°F OAT Switchover', '40°F OAT Switchover', 'Cost Switchover']
    nh_colors   = ['#7030A0', '#0070C0', '#C00000']
    nh_ls       = [(0,(4,2)), (0,(2,2)), (0,(1,1))]
    nh_sys      = ['SS HP', 'Two-stage HP', 'Modulating HP']

    n_cfg      = len(HYBRID_OPTIONS)
    x_hyb      = np.arange(3*n_cfg)
    div1, div2 = n_cfg-0.5, 2*n_cfg-0.5
    x_nh_base  = 3*n_cfg + 1.2
    x_nh       = [x_nh_base + i for i in range(len(nh_sys))]
    x_labels   = ([cfg.replace('NC Furnace + ','NC\nFurnace +\n')
                       .replace('Cond Furnace + ','Cond\nFurnace +\n')
                   for cfg in HYBRID_OPTIONS] * 3 + nh_sys)
    sec_bg     = ['lightyellow', 'lightgoldenrodyellow', '#e8f4e8']

    def peak_oat(sc, cfg, thr):
        hc  = sizing_df.loc[sc, 'Cooling_Oversize_Selected_MBH']
        fc  = sizing_df.loc[sc, 'Furnace_Selected_MBH']
        fe  = 0.80 if cfg.startswith('NC') else 0.95
        OAT = np.array(load_dfs[sc]['OAT F'])
        L   = np.array(load_dfs[sc]['Heating MBH'])
        ht  = _hp_type_from_cfg(cfg)
        _, HE, _, _ = _run_hp(ht, hc, OAT, L, eq)
        _, FE, _    = sim_furnace(fc, fe, L)
        return np.where(OAT >= thr, HE, FE).max()

    plt.rcParams.update(_RC)
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.subplots_adjust(hspace=0.25)

    for ax, sc in zip(axes, [sc_old, sc_new]):
        if sc not in load_dfs:
            ax.set_title(f'{sc} — data not available'); continue

        hyb_peaks = []; hyb_colors = []
        for thr, col in zip([35, 40], ctrl_colors[:2]):
            for cfg in HYBRID_OPTIONS:
                hyb_peaks.append(peak_oat(sc, cfg, thr)); hyb_colors.append(col)
        for cfg in HYBRID_OPTIONS:
            key = (f"{sc}__{hp_strat}__{cfg}__Cost_Switchover"
                   .replace(' ','_').replace('+','and'))
            df_c = hourly_results[key]
            hyb_peaks.append((df_c['HP_Elec_kWh']+df_c['Furnace_Elec_kWh']).max())
            hyb_colors.append(ctrl_colors[2])
        hyb_peaks = np.array(hyb_peaks)

        nh_peaks = []
        for sys in nh_sys:
            key = f"{sc}__{NH_HP_STRATEGY}__{sys}".replace(' ','_')
            nh_peaks.append(hourly_results[key]['Elec_kWh'].max())

        ymax = max(np.concatenate([hyb_peaks, nh_peaks]))

        ax.bar(x_hyb, hyb_peaks, width=0.6, color=hyb_colors, edgecolor='white', zorder=2)
        for xi, p in zip(x_hyb, hyb_peaks):
            ax.text(xi, p+ymax*0.01, f'{p:.1f}', ha='center', va='bottom', rotation=90, fontsize=7)
        ax.bar(x_nh, nh_peaks, width=0.6, color=nh_colors, edgecolor='white', zorder=2, alpha=0.85)
        for xi, p in zip(x_nh, nh_peaks):
            ax.text(xi, p+ymax*0.01, f'{p:.1f}', ha='center', va='bottom', rotation=90, fontsize=7)
        for pv, col, ls, sys in zip(nh_peaks, nh_colors, nh_ls, nh_sys):
            ax.hlines(pv, xmin=-0.5, xmax=x_hyb[-1]+0.5,
                      colors=col, linestyles=ls, linewidth=1.8, alpha=0.75)
            ax.text(x_hyb[-1]+0.55, pv, f' ← {sys.replace(" HP","")} standalone',
                    ha='left', va='center', fontsize=8, color=col, fontstyle='italic')

        ax.axvspan(-0.5,    div1,           color=sec_bg[0], alpha=0.35, zorder=0)
        ax.axvspan(div1,    div2,           color=sec_bg[1], alpha=0.55, zorder=0)
        ax.axvspan(div2,    x_hyb[-1]+0.5, color=sec_bg[2], alpha=0.35, zorder=0)
        ax.axvspan(x_hyb[-1]+0.5, x_nh[-1]+0.5, color='#f0f0f0', alpha=0.6, zorder=0)
        for d in [div1, div2, x_hyb[-1]+0.85]:
            ax.axvline(x=d, color='black', linestyle='--', linewidth=1, alpha=0.5)

        ax.text((-0.5+div1)/2,          ymax*1.18, ctrl_labels[0], ha='center', fontsize=9, color='dimgray', style='italic')
        ax.text((div1+div2)/2,           ymax*1.18, ctrl_labels[1], ha='center', fontsize=9, color='dimgray', style='italic')
        ax.text((div2+x_hyb[-1]+0.5)/2, ymax*1.18, ctrl_labels[2], ha='center', fontsize=9, color='dimgray', style='italic')
        ax.text(np.mean(x_nh),           ymax*1.18, 'All-Electric\n(Non-Hybrid)', ha='center', fontsize=9, color='dimgray', style='italic')

        ax.set_ylabel('Peak Electrical Demand (kW)');  ax.set_ylim(0, ymax*1.30)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7);  ax.set_axisbelow(True)
        ax.set_title(sc_labels[sc], fontsize=11, fontweight='bold')
        legend_h = ([mpatches.Patch(facecolor=c, label=l) for c, l in zip(ctrl_colors, ctrl_labels)] +
                    [Line2D([0],[0], color=c, linestyle=ls, linewidth=1.8, label=f'{s} standalone')
                     for c, ls, s in zip(nh_colors, nh_ls, nh_sys)])
        ax.legend(handles=legend_h, loc='upper left', fontsize=8, ncol=2, framealpha=0.9)

    all_xs = list(x_hyb) + x_nh
    axes[-1].set_xticks(all_xs)
    axes[-1].set_xticklabels(x_labels, rotation=45, ha='center', fontsize=8)
    plt.tight_layout()
    return fig


# ── Plot: City-Level ──────────────────────────────────────────────────────────
def plot_city_level(total_units, housing_fracs, results, non_hybrid_results, hourly_results):
    unit_counts = {s: round(f * total_units) for s, f in housing_fracs.items()}
    HYB_STRAT   = 'Sized for Cooling'

    systems = {
        'NC\nFurnace':     ('NH','NC Furnace',                   None),
        'Cond\nFurnace':   ('NH','Cond Furnace',                 None),
        'NC+SS\n35F':      ('HY','NC Furnace + SS HP',           '35F Switchover'),
        'NC+SS\n40F':      ('HY','NC Furnace + SS HP',           '40F Switchover'),
        'NC+SS\nCost':     ('HY','NC Furnace + SS HP',           'Cost Switchover'),
        'NC+Mod\n35F':     ('HY','NC Furnace + Modulating HP',   '35F Switchover'),
        'NC+Mod\n40F':     ('HY','NC Furnace + Modulating HP',   '40F Switchover'),
        'NC+Mod\nCost':    ('HY','NC Furnace + Modulating HP',   'Cost Switchover'),
        'Cond+Mod\n35F':   ('HY','Cond Furnace + Modulating HP', '35F Switchover'),
        'Cond+Mod\n40F':   ('HY','Cond Furnace + Modulating HP', '40F Switchover'),
        'Cond+Mod\nCost':  ('HY','Cond Furnace + Modulating HP', 'Cost Switchover'),
        'Cond+SS\n35F':    ('HY','Cond Furnace + SS HP',         '35F Switchover'),
        'Cond+SS\n40F':    ('HY','Cond Furnace + SS HP',         '40F Switchover'),
        'Cond+SS\nCost':   ('HY','Cond Furnace + SS HP',         'Cost Switchover'),
        'All-Elec\n(SS)':  ('NH','SS HP',                        None),
        'All-Elec\n(Mod)': ('NH','Modulating HP',                None),
    }

    ann_gas = {};  ann_elec = {};  pk_gas = {};  pk_elec = {}
    for lbl, (stype, cfg, ctrl) in systems.items():
        ag=0.0; ae=0.0; hg=np.zeros(8760); he=np.zeros(8760)
        for sc, n in unit_counts.items():
            if stype == 'NH':
                r   = non_hybrid_results[sc][NH_HP_STRATEGY][cfg]
                ag += r['Gas_MBH']*n;  ae += r['Electricity_kWh']*n
                key = f"{sc}__{NH_HP_STRATEGY}__{cfg}".replace(' ','_')
                df_h = hourly_results[key]
                hg  += df_h['Gas_MBH'].values*n;  he += df_h['Elec_kWh'].values*n
            else:
                r   = results[sc][HYB_STRAT][cfg][ctrl]
                ag += r['Gas_MBH']*n;  ae += r['Electricity_kWh']*n
                key = (f"{sc}__{HYB_STRAT}__{cfg}__{ctrl}"
                       .replace(' ','_').replace('+','and'))
                df_h = hourly_results[key]
                hg  += df_h['Furnace_Gas_MBH'].values*n
                he  += (df_h['HP_Elec_kWh']+df_h['Furnace_Elec_kWh']).values*n
        ann_gas[lbl]  = ag/1000;  ann_elec[lbl] = ae/1000
        pk_gas[lbl]   = hg.max()/1000;  pk_elec[lbl] = he.max()/1000

    labels     = list(systems.keys())
    bar_colors = ['#C55A11','#E2975A',
                  '#70AD47','#538135','#375623',
                  '#4BACC6','#2E75B6','#1F4E79',
                  '#9DC3E6','#5B9BD5','#2E75B6',
                  '#C896C8','#A64CA6','#7B2D7B',
                  '#FF0000','#C00000']
    x = np.arange(len(labels));  w = 0.65
    gas_end=1.5;  hyb_end=13.5;  sub_divs=[4.5,7.5,10.5]

    plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(2, 2, figsize=(22, 10))
    panels = [
        (axes[0,0], [ann_gas[l]  for l in labels], 'Annual Gas (Thousand MMBTU/yr)',  'GAS — Annual Supply Planning'),
        (axes[0,1], [ann_elec[l] for l in labels], 'Annual Electricity (GWh/yr)',      'ELECTRIC — Annual Supply Planning'),
        (axes[1,0], [pk_gas[l]   for l in labels], 'Peak Gas Demand (MMBTU/hr)',       'GAS — Pipeline / Distribution Capacity'),
        (axes[1,1], [pk_elec[l]  for l in labels], 'Peak Electric Demand (MW)',        'ELECTRIC — Generation / Grid Capacity'),
    ]
    for ax, vals, ylabel, title in panels:
        bars = ax.bar(x, vals, width=w, color=bar_colors, edgecolor='white', zorder=2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.008,
                    f'{v:,.1f}', ha='center', va='bottom', fontsize=7, rotation=90)
        ax.axvspan(-0.5,    gas_end,         color='#FFF0E8', alpha=0.55, zorder=0)
        ax.axvspan(gas_end, hyb_end,         color='#F0FFF0', alpha=0.45, zorder=0)
        ax.axvspan(hyb_end, len(labels)-0.5, color='#E8F0FF', alpha=0.55, zorder=0)
        for d in [gas_end, hyb_end]: ax.axvline(d, color='black', linestyle='--', linewidth=1, alpha=0.5)
        for d in sub_divs:           ax.axvline(d, color='gray',  linestyle=':',  linewidth=0.9, alpha=0.6)
        ymax = max(vals)
        ax.set_xticks(x);  ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=9);  ax.set_title(title, fontsize=10, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.4);  ax.set_axisbelow(True)
        ax.set_ylim(0, ymax*1.28)
        ax.text(0.5,  ymax*1.21, 'Gas\nOnly',    ha='center', fontsize=8, color='#C55A11', fontstyle='italic')
        ax.text(3.0,  ymax*1.21, 'NC+SS HP',     ha='center', fontsize=7, color='#375623', fontstyle='italic')
        ax.text(6.0,  ymax*1.21, 'NC+Mod HP',    ha='center', fontsize=7, color='#1F4E79', fontstyle='italic')
        ax.text(9.0,  ymax*1.21, 'Cond+Mod HP',  ha='center', fontsize=7, color='#2E75B6', fontstyle='italic')
        ax.text(12.0, ymax*1.21, 'Cond+SS HP',   ha='center', fontsize=7, color='#7B2D7B', fontstyle='italic')
        ax.text(14.5, ymax*1.21, 'All\nElec',    ha='center', fontsize=8, color='#C00000', fontstyle='italic')

    fig.suptitle(f'Birmingham City-Level Utility Planning  |  {total_units:,} units',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.title("Hybrid Heating System Analysis")
    st.caption("Spire — Birmingham, AL  |  BeOpt loads × Carrier ASHP performance")

    # ── File uploads ──
    st.sidebar.header("Data Files")
    st.sidebar.caption("Upload the three Excel files to get started.")
    f_load     = st.sidebar.file_uploader("Load_Data.xlsx",              type="xlsx", key="load")
    f_twostage = st.sidebar.file_uploader("60MBH_TwoStage_Data.xlsx",    type="xlsx", key="two")
    f_carrier  = st.sidebar.file_uploader("60MBH_Carrier_ASHP_Data.xlsx", type="xlsx", key="carrier")

    if not (f_load and f_twostage and f_carrier):
        st.info("Upload all three Excel files in the sidebar to begin.")
        st.stop()

    load_bytes     = f_load.getvalue()
    twostage_bytes = f_twostage.getvalue()
    carrier_bytes  = f_carrier.getvalue()

    # ── Parameters ──
    st.sidebar.header("Parameters")
    st.sidebar.subheader("Energy Prices")
    elec_price = st.sidebar.number_input("Electricity ($/kWh)",  0.01, 1.0,   0.14,   step=0.005,  format="%.4f")
    gas_rate   = st.sidebar.number_input("Natural Gas ($/MBH)",  0.001, 0.10, 0.0134, step=0.0005, format="%.5f")
    st.sidebar.subheader("Furnace Efficiency")
    nc_eff   = st.sidebar.slider("Non-Condensing (%)", 70, 85, 80, step=1) / 100.0
    cond_eff = st.sidebar.slider("Condensing (%)",     88, 98, 95, step=1) / 100.0

    results, non_hybrid_results, hourly_results, sizing_df = run_simulation(
        elec_price, gas_rate, nc_eff, cond_eff, load_bytes, twostage_bytes, carrier_bytes)
    load_dfs, eq = load_all_data(load_bytes, twostage_bytes, carrier_bytes)

    tab1, tab2, tab3, tab4 = st.tabs(["Annual Cost", "Annual Energy", "Peak Demand", "City-Level Analysis"])

    with tab1:
        c1, c2 = st.columns([1, 3])
        with c1:
            sc_cost   = st.selectbox("Scenario", BAL_SCENARIOS, key='sc_cost')
            ctrl_cost = st.multiselect("Control strategies", CONTROL_STRATEGIES,
                                        default=CONTROL_STRATEGIES, key='ctrl_cost')
        with c2:
            if ctrl_cost:
                fig = plot_annual_cost(sc_cost, ctrl_cost, results, non_hybrid_results)
                show_fig(fig)
            else:
                st.info("Select at least one control strategy.")

    with tab2:
        c1, c2 = st.columns([1, 3])
        with c1:
            sc_ene   = st.selectbox("Scenario", BAL_SCENARIOS, key='sc_ene')
            ctrl_ene = st.multiselect("Control strategies", CONTROL_STRATEGIES,
                                       default=['35F Switchover','40F Switchover','Cost Switchover'],
                                       key='ctrl_ene')
        with c2:
            if ctrl_ene:
                fig = plot_annual_energy(sc_ene, ctrl_ene, results, non_hybrid_results)
                show_fig(fig)
            else:
                st.info("Select at least one control strategy.")

    with tab3:
        c1, c2 = st.columns([1, 3])
        with c1:
            home_size = st.radio("Home size (sqft)", [1800, 2400, 3000], index=1, key='home_sz')
        with c2:
            fig = plot_peak_demand(home_size, results, hourly_results, sizing_df, eq, load_dfs)
            show_fig(fig)

    with tab4:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.subheader("Housing Stock")
            total_units = st.number_input("Total residential units", 10_000, 200_000, 63_600, step=1_000)
            st.markdown("**Fractions by segment**")
            fracs_default = {
                'BAL_PCV_1800': 0.48, 'BAL_PCV_2400': 0.18, 'BAL_PCV_3000': 0.06,
                'BAL_IECC_1800': 0.04,'BAL_IECC_2400': 0.13,'BAL_IECC_3000': 0.11,
            }
            housing_fracs = {}; total_frac = 0.0
            for seg, default in fracs_default.items():
                v = st.number_input(seg, 0.0, 1.0, default, step=0.01, format="%.2f", key=f'frac_{seg}')
                housing_fracs[seg] = v;  total_frac += v
            if abs(total_frac - 1.0) > 0.01:
                st.warning(f"Fractions sum to {total_frac:.2f} — should be 1.00")
        with c2:
            fig = plot_city_level(total_units, housing_fracs, results, non_hybrid_results, hourly_results)
            show_fig(fig)

    with st.expander("Equipment Sizing Results"):
        st.dataframe(sizing_df.style.format({
            col: '{:.1f}' for col in sizing_df.select_dtypes('float').columns
        }), use_container_width=True)


if __name__ == "__main__":
    main()
