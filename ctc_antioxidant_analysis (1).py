"""CanTreatCOVID antioxidant arm (PLOS ONE PONE-D-26-28075, R1): analysis code.

Reproduces every number in the revised manuscript that comes from the trial data:
Table 1 (except comorbidity rows), feasibility and safety, Table 4 and S3 Table (Bayesian models),
Table 2 and S1 Table (dietary recalls), Table 3 and S2 Table (supplement intake).

Inputs (place in DATA_DIR):
  CTC_data_antiox_202609.xlsx                          REDCap export, one row per participant-event
  FoodDiaryAnalysis_CanTreatCovid_251218.xlsx          24-hour recall totals, Day 0 and Day 10
  food_data_AO_VitaminsDietarySupplements_Part2.xlsx   supplement totals, sheet 'TOTALS - last 12 mos'
Usage:  python ctc_antioxidant_analysis.py [DATA_DIR] [OUT_DIR]
Requires: Python 3.10+, numpy, scipy, pandas, openpyxl. Run time about 5 minutes (MCMC).

Analysis population: participants randomized to antioxidant therapy (n=40) or usual care (n=40).
Participants randomized to the nirmatrelvir/ritonavir arm are excluded (3-1004 in the trial file;
6-1001 in the food files).
"""
import itertools, json, os, re, sys
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else 'results'
os.makedirs(OUT_DIR, exist_ok=True)
F_TRIAL = os.path.join(DATA_DIR, 'CTC_data_antiox_202609.xlsx')
F_DIARY = os.path.join(DATA_DIR, 'FoodDiaryAnalysis_CanTreatCovid_251218.xlsx')
F_SUPP = os.path.join(DATA_DIR, 'food_data_AO_VitaminsDietarySupplements_Part2.xlsx')
SYM = ['fever', 'cough', 'sob', 'taste', 'muscle_ache', 'nausea', 'fatigue1', 'concetrate', 'mood']


# =============================================================== 1. trial data
def load_trial():
    df = pd.read_excel(F_TRIAL, dtype=object)
    df = df[df.participant_id.notna()]
    ev = df.redcap_event_name.fillna('')
    P = df[~ev.str.contains('Diary|Diaries')].groupby('participant_id').first()   # participant level
    rows = []
    for pat, pre in [(r'^Daily e-Diary\((\d+)\)$', 'pdd_'), (r'^Flu Pro Diaries\((\d+)\)$', 'fpp_')]:
        for _, r in df[ev.str.match(pat)].iterrows():
            rec = {'pid': r.participant_id, 'day': int(re.match(pat, r.redcap_event_name).group(1))}
            for k in ['recover', 'return_health', 'return_activity', 'feel_today', 'hospital'] + SYM:
                rec[k] = pd.to_numeric(r.get(pre + k), errors='coerce')
            rows.append(rec)
    D = pd.DataFrame(rows).sort_values(['pid', 'day'])
    D['answered'] = D[['recover', 'return_health', 'return_activity', 'feel_today']].notna().any(axis=1)
    return df, P, D


def derive_outcomes(P, D):
    """Day-14 diary outcomes (NaN = no answered diary day) and hospitalization/death by day 28."""
    Dd = D[D.answered].copy()
    Dd['ok'] = Dd.feel_today.le(1)              # global rating: no (0) or mild (1) symptoms

    def sustained(g):                            # 3 consecutive days no/mild, no later rating >= moderate
        m, days = g.ok.values, g.day.values
        for i in range(len(m) - 2):
            if m[i] and m[i + 1] and m[i + 2] and days[i + 2] - days[i] == 2 and m[i:].all():
                return True
        return False
    g = Dd.groupby('pid')
    O = pd.DataFrame({'rec': g.recover.apply(lambda s: (s == 1).any()),
                      'hea': g.return_health.apply(lambda s: (s == 1).any()),
                      'act': g.return_activity.apply(lambda s: (s == 1).any()),
                      'all': g.ok.any(),
                      'sus': g.apply(sustained)}).astype(float)
    O = O.reindex(P.index)
    hosp = pd.Series(0.0, index=P.index)
    hosp[D[D.hospital == 1].pid.unique()] = 1.0
    for c in ['fup_cov_hosp', 'fup28_cov_hosp', 'er1_hospitalized(1)']:
        hosp[pd.to_numeric(P[c], errors='coerce') == 1] = 1.0
    hosp[P.death_date.notna()] = 1.0
    O['hosp'] = hosp
    return O


def covariates(df, P):
    A = pd.DataFrame(index=P.index)
    A['arm'] = P.rand_group
    dob = pd.to_datetime(P.dem_dob.replace('{null}', np.nan), errors='coerce')
    A['age'] = ((pd.to_datetime(P.rand_date) - dob).dt.days / 365.25).round(1)
    A['male'] = P.dem_sex.map({1: 1.0, 2: 0.0})
    A['vacc2'] = pd.to_numeric(P.dem_vaccination_status, errors='coerce').map({2: 1.0, 1: 0.0, 0: 0.0})
    A['comorb'] = pd.to_numeric(P.chronic_disease, errors='coerce').astype(float)
    # nirmatrelvir/ritonavir recorded in concomitant medications, overlapping follow-up
    names = [c for c in P.columns if re.match(r'med_name\(\d+\)', c)]
    pax_rows = []
    for pid in P.index:
        for c in names:
            v = P.loc[pid, c]
            if pd.notna(v) and re.search(r'paxlo|nirmat|ritonav', str(v), re.I):
                i = re.search(r'\((\d+)\)', c).group(1)
                start = pd.to_datetime(P.loc[pid, f'med_start({i})'], errors='coerce')
                pax_rows.append({'pid': pid, 'arm': A.loc[pid, 'arm'],
                                 'start_rel_rand': (start - pd.to_datetime(P.loc[pid, 'rand_date'])).days})
    pax = pd.DataFrame(pax_rows).drop_duplicates('pid')
    A['pax'] = A.index.isin(pax.pid).astype(float)
    return A, pax


# =============================================================== 2. Bayesian logistic regression
def _log_t(x, df, loc, scale):
    z = (x - loc) / scale
    return (special.gammaln((df + 1) / 2) - special.gammaln(df / 2) - 0.5 * np.log(df * np.pi)
            - np.log(scale) - (df + 1) / 2 * np.log1p(z * z / df))


class Model:
    """y ~ Bernoulli(logit^-1(a + b_trt*trt + X b)); b ~ N(0, sd_beta); a ~ Student-t(1, mu0, 2.5).
    Missing binary covariates: summed out, x ~ Bernoulli(pi), pi ~ Beta(1,1).
    Missing continuous covariates: mean-imputed (standardized scale)."""

    def __init__(self, y, trt, X, binary, mu0, sd_beta=1.0):
        self.y, self.trt = np.asarray(y, float), np.asarray(trt, float)
        X = np.asarray(X, float).copy() if X.size else np.zeros((len(y), 0))
        self.k, self.binary = X.shape[1], list(binary)
        for j in range(self.k):
            if not self.binary[j]:
                X[np.isnan(X[:, j]), j] = np.nanmean(X[:, j])
        self.X, self.mu0, self.sd_beta = X, mu0, sd_beta
        self.mis = [j for j in range(self.k) if self.binary[j] and np.isnan(X[:, j]).any()]
        self.patterns = {}
        for i in range(len(self.y)):
            self.patterns.setdefault(tuple(j for j in self.mis if np.isnan(X[i, j])), []).append(i)
        self.dim = 2 + self.k + len(self.mis)

    def logpost(self, th):
        th = np.atleast_2d(th)
        a, bt, B, lpi = th[:, 0], th[:, 1], th[:, 2:2 + self.k], th[:, 2 + self.k:]
        lp = _log_t(a, 1.0, self.mu0, 2.5) + stats.norm.logpdf(bt, 0, self.sd_beta)
        if self.k:
            lp = lp + stats.norm.logpdf(B, 0, self.sd_beta).sum(axis=1)
        if lpi.shape[1]:
            lp = lp + (-np.logaddexp(0, -lpi) - np.logaddexp(0, lpi)).sum(axis=1)
        pis = special.expit(lpi)
        for miss, idx in self.patterns.items():
            idx = np.array(idx)
            base, y, t = np.nan_to_num(self.X[idx].copy()), self.y[idx], self.trt[idx]
            acc = None
            for combo in itertools.product([0, 1], repeat=len(miss)):
                Xc, logw = base.copy(), np.zeros(th.shape[0])
                for v, j in zip(combo, miss):
                    Xc[:, j] = v
                    pj = pis[:, self.mis.index(j)]
                    logw = logw + (np.log(pj) if v else np.log1p(-pj))
                eta = a[:, None] + bt[:, None] * t[None, :] + B @ Xc.T
                term = y[None, :] * eta - np.logaddexp(0, eta) + logw[:, None]
                acc = term if acc is None else np.logaddexp(acc, term)
            lp = lp + acc.sum(axis=1)
        return lp

    def sample(self, n_iter=30000, burn=8000, chains=8, seed=1, thin=4):
        """Adaptive random-walk Metropolis (Haario et al. 2001), chains run in parallel."""
        rng = np.random.default_rng(seed)
        x0 = np.zeros(self.dim); x0[0] = self.mu0
        opt = optimize.minimize(lambda t: -self.logpost(t[None, :])[0], x0, method='BFGS')
        H = np.atleast_2d(opt.hess_inv); H = (H + H.T) / 2 + 1e-6 * np.eye(self.dim)
        L = np.linalg.cholesky(H)
        cur = opt.x + (L @ rng.standard_normal((self.dim, chains))).T * 1.5
        cur_lp, scale, hist, draws = self.logpost(cur), 2.38 / np.sqrt(self.dim), [], []
        for it in range(n_iter):
            if it > 0 and it % 2000 == 0 and it <= burn:
                L = np.linalg.cholesky(np.cov(np.concatenate(hist[-2000:]).T) + 1e-6 * np.eye(self.dim))
            prop = cur + scale * (L @ rng.standard_normal((self.dim, chains))).T
            plp = self.logpost(prop)
            acc = np.log(rng.random(chains)) < plp - cur_lp
            cur, cur_lp = np.where(acc[:, None], prop, cur), np.where(acc, plp, cur_lp)
            if it < burn:
                hist.append(cur.copy())
            elif (it - burn) % thin == 0:
                draws.append(cur.copy())
        self.draws = np.stack(draws, axis=1)          # (chains, draws, dim)
        return self.draws


def rhat_ess(x):
    c, n = x.shape; h = n // 2
    s = np.concatenate([x[:, :h], x[:, h:2 * h]]); m, nn = s.shape
    W = s.var(axis=1, ddof=1).mean(); B = nn * s.mean(axis=1).var(ddof=1)
    rhat = np.sqrt(((nn - 1) / nn * W + B / nn) / W)
    acf = np.zeros(nn)
    for ch in s:
        z = ch - ch.mean(); f = np.fft.rfft(z, 2 * nn); ac = np.fft.irfft(f * np.conj(f))[:nn]; acf += ac / ac[0]
    acf /= m; tau = 1.0
    for t in range(1, nn - 1, 2):
        if acf[t] + acf[t + 1] < 0: break
        tau += 2 * (acf[t] + acf[t + 1])
    return float(rhat), float(m * nn / tau)


def fit(A, outcome, covs, sd_beta=1.0, seed=1, data=None):
    d = (A if data is None else data)
    d = d[d.arm.isin(['Antioxidant', 'Usual Care']) & d[outcome].notna()]
    trt = (d.arm == 'Antioxidant').astype(float).values
    X, binary = [], []
    for c in covs:
        v = d[c].astype(float).values.copy()
        if c == 'age':
            v = (v - np.nanmean(v)) / np.nanstd(v); binary.append(False)
        else:
            binary.append(True)
        X.append(v)
    X = np.column_stack(X) if X else np.zeros((len(d), 0))
    mu0 = special.logit(0.03) if outcome == 'hosp' else 0.0
    m = Model(d[outcome].values, trt, X, binary, mu0=mu0, sd_beta=sd_beta)
    bt = m.sample(seed=seed)[:, :, 1]
    orr = np.exp(bt.ravel())
    better = (bt.ravel() < 0) if outcome == 'hosp' else (bt.ravel() > 0)
    rh, ess = rhat_ess(bt)
    y = d[outcome].values
    return {'n_trt': int(trt.sum()), 'ev_trt': int(y[trt == 1].sum()), 'n_ctl': int((1 - trt).sum()), 'ev_ctl': int(y[trt == 0].sum()),
            'OR_median': float(np.median(orr)), 'OR_mean': float(orr.mean()),
            'CrI_lo': float(np.quantile(orr, .025)), 'CrI_hi': float(np.quantile(orr, .975)),
            'P_superiority': float(better.mean()), 'rhat': rh, 'ess': ess}


# =============================================================== 3. dietary recalls and supplements
S1_NUTRIENTS = [ "18:2 - Linoleic (g)", "18:3 - Linolenic (g)", "20:3 - Eicosatrienoic (g)", "20:4 - Arachidon (g)", "20:5 - EPA (g)", "22:5 - DPA (g)", "22:6 - DHA (g)", "Alpha-Carotene (mcg)", "Beta-Carotene Equiv (mcg)", "Biotin (mcg)", "Caffeine (mg)", "Calcium (mg)", "Calories (kcal)", "Calories from Fat (kcal)", "Calories from SatFat (kcal)", "Calories from TransFat (kcal)", "Carbohydrates (g)", "Carotenoid RE (mcg)", "Choline (mg)", "Fat (g)", "Folate (mcg)", "Folate, DFE (mcg DFE)", "Folate, food (mcg)", "Folic Acid (mcg)", "Glycemic Load", "Iodine (mcg)", "Iron (mg)", "Lutein & Zeaxanthin (mcg)", "Lycopene (mcg)", "Magnesium (mg)", "Mono Fat (g)", "MyPlate - Dairy (c)", "MyPlate - Fruit (c)", "MyPlate - Grain Total (oz-eq)", "MyPlate - Protein Total (oz-eq)", "MyPlate - Vegetable Total (c)", "Net Carbs (g)", "Omega 3 Fatty Acid (g)", "Omega 6 Fatty Acid (g)", "Pantothenic Acid (mg)", "Phosphorus (mg)", "Poly Fat (g)", "Potassium (mg)", "Protein (g)", "Saturated Fat (g)", "Selenium (mcg)", "Sodium (mg)", "Total Dietary Fiber (g)", "Total Soluble Fiber (g)", "Trans Fatty Acid (g)", "Vitamin A - IU (IU)", "Vitamin B1 - Thiamin (mg)", "Vitamin B12 (mcg)", "Vitamin B2 - Riboflavin (mg)", "Vitamin B3 - Niacin Equiv (mg)", "Vitamin B6 (mg)", "Vitamin C (mg)", "Vitamin D - IU (IU)", "Vitamin E - IU (IU)", "Vitamin K (mcg)", "Zinc (mg)" ]   # the 61 nutrients in S1 Table


def dietary(arm, nutrients):
    fd = pd.read_excel(F_DIARY)
    fd['pid'] = fd['First Name'].str.replace(r'(?i)cantreatcovid', '', regex=True).str.strip()
    fd['day'] = fd['Last Name'].str.extract(r'(\d+)').astype(int)
    fd = pd.concat([fd[['pid', 'day']], fd[nutrients].apply(pd.to_numeric, errors='coerce')], axis=1)
    ids = {g: list(arm.index[arm == lab]) for g, lab in [('ao', 'Antioxidant'), ('uc', 'Usual Care')]}
    out = []
    for n in nutrients:
        row, W = {'nutrient': n}, {}
        for g, pid in ids.items():
            w = fd[fd.pid.isin(pid)].pivot(index='pid', columns='day', values=n).dropna()
            W[g] = w
            ch = w[10] - w[0]; se = ch.std(ddof=1) / np.sqrt(len(ch)); q = stats.t.ppf(.975, len(ch) - 1)
            row.update({f'{g}_n': len(w), f'{g}_base_mean': w[0].mean(), f'{g}_base_sd': w[0].std(),
                        f'{g}_d10_mean': w[10].mean(), f'{g}_d10_sd': w[10].std(),
                        f'{g}_change': ch.mean(), f'{g}_change_lo': ch.mean() - q * se, f'{g}_change_hi': ch.mean() + q * se,
                        f'{g}_p': stats.ttest_rel(w[10], w[0]).pvalue})
        y = np.r_[W['ao'][10].values, W['uc'][10].values]; b = np.r_[W['ao'][0].values, W['uc'][0].values]
        t = np.r_[np.ones(len(W['ao'])), np.zeros(len(W['uc']))]
        Xm = np.column_stack([np.ones_like(y), t, b]); beta = np.linalg.lstsq(Xm, y, rcond=None)[0]
        res = y - Xm @ beta; dfree = len(y) - 3; cov = res @ res / dfree * np.linalg.pinv(Xm.T @ Xm)
        se = np.sqrt(cov[1, 1]); q = stats.t.ppf(.975, dfree)
        row.update({'beta': beta[1], 'beta_lo': beta[1] - q * se, 'beta_hi': beta[1] + q * se,
                    'beta_p': 2 * stats.t.sf(abs(beta[1] / se), dfree)})
        yr, br = pd.Series(y).rank().values, pd.Series(b).rank().values          # rank-based ANCOVA
        Xr = np.column_stack([np.ones_like(yr), t, br]); gr = np.linalg.lstsq(Xr, yr, rcond=None)[0]; rr = yr - Xr @ gr
        ser = np.sqrt(rr @ rr / dfree * np.linalg.pinv(Xr.T @ Xr)[1, 1])
        row['rank_ancova_p'] = 2 * stats.t.sf(abs(gr[1] / ser), dfree)
        out.append(row)
    return pd.DataFrame(out)


def supplements(arm):
    tot = pd.read_excel(F_SUPP, sheet_name='TOTALS - last 12 mos').rename(columns={'Vit/Min (per day)': 'pid'})
    tot['pid'] = tot.pid.astype(str).str.strip()
    tot['arm'] = tot.pid.map(arm)
    tot = tot[tot.arm.isin(['Antioxidant', 'Usual Care'])]
    cols = [c for c in tot.columns if c not in ('Order', 'pid', 'arm')]
    num = tot[cols].apply(pd.to_numeric, errors='coerce')
    return pd.DataFrame({f'{g}_{s}': getattr(num[tot.arm == lab], s)() for g, lab in [('ao', 'Antioxidant'), ('uc', 'Usual Care')]
                         for s in ['mean', 'std']}).rename_axis('nutrient')


# =============================================================== 4. main
def cp(k, n):
    return [k, n, 100 * k / n, 100 * (stats.beta.ppf(.025, k, n - k + 1) if k else 0), 100 * (stats.beta.ppf(.975, k + 1, n - k) if k < n else 1)]


def main():
    df, P, D = load_trial()
    arm = P.rand_group
    O = derive_outcomes(P, D)
    A, pax = covariates(df, P)
    A = A.join(O)
    AO, UC = list(arm.index[arm == 'Antioxidant']), list(arm.index[arm == 'Usual Care'])
    R = {'n_randomized': arm.value_counts().to_dict()}

    # ---- Table 1 (rows that can be derived directly; comorbidity rows need the statisticians' grouping)
    bl = df[df.redcap_event_name == 'Baseline'].set_index('participant_id')
    age = pd.to_numeric(P.dem_age_calc, errors='coerce').copy()
    age[pd.to_datetime(P.dem_dob.replace('{null}', np.nan), errors='coerce').notna() & (age == 0)] = np.nan
    age = age.fillna(A.age.round(0))           # 1-1002: dem_age_calc = 0, date of birth gives 69
    t1 = {}
    for g, ids in [('antioxidant', AO), ('usual_care', UC)]:
        a = age.reindex(ids).dropna(); sex = P.loc[ids, 'dem_sex']; race = pd.to_numeric(P.loc[ids, 'dem_race'], errors='coerce')
        dur = (pd.to_datetime(bl.reindex(ids)['visit_date'], errors='coerce') - pd.to_datetime(P.loc[ids, 'symp_onset_date'])).dt.days
        bmi = pd.to_numeric(P.loc[ids, 'bmi'], errors='coerce')
        S = P.loc[ids, ['covid_' + s for s in SYM]].apply(pd.to_numeric, errors='coerce')
        t1[g] = {'age_mean_sd_min_max': [a.mean(), a.std(), a.min(), a.max()],
                 'male': int((sex == 1).sum()), 'female': int((sex == 2).sum()), 'sex_missing': int(sex.isna().sum()),
                 'white': int((race == 1).sum()), 'asian': int(race.isin([4, 6, 7, 8]).sum()), 'indigenous': int((race == 5).sum()),
                 'mixed': int((race == 9).sum()), 'ethnicity_missing': int(race.isna().sum()),
                 'symptom_days_mean_sd_median_q1_q3': [dur.mean(), dur.std(), dur.median(), dur.quantile(.25), dur.quantile(.75)],
                 'vaccine_doses_0_1_2plus_missing': [int((pd.to_numeric(P.loc[ids, 'dem_vaccination_status'], errors='coerce') == k).sum()) for k in (0, 1, 2)]
                                                    + [int(P.loc[ids, 'dem_vaccination_status'].isna().sum())],
                 'bmi_median_q1_q3': [bmi.median(), bmi.quantile(.25), bmi.quantile(.75)],
                 'symptoms_no_mild_moderate_major_missing': {s: [int((S['covid_' + s] == k).sum()) for k in range(4)] + [int(S['covid_' + s].isna().sum())] for s in SYM},
                 'any_symptom_moderate_or_major': int((S >= 2).any(axis=1)[S.notna().any(axis=1)].sum())}
    R['table1'] = t1

    # ---- feasibility and safety
    end = pd.to_numeric(P.end_study_reason, errors='coerce')      # 5 = completed, 4 = lost to follow-up, 6 = withdrew
    R['retention'] = {'antioxidant': cp(int((end[AO] == 5).sum()), 40), 'usual_care': cp(int((end[UC] == 5).sum()), 40)}
    R['adherence_antioxidant'] = cp(int((pd.to_numeric(P.loc[AO, 'end_treat_yn'], errors='coerce') == 1).sum()), 40)
    ae_any = P.ae_yn.map(lambda v: v is True or v == 1)
    R['adverse_events_participants'] = {'antioxidant': cp(int(ae_any[AO].sum()), 40), 'usual_care': cp(int(ae_any[UC].sum()), 40)}
    R['adverse_events'] = [{'arm': arm[p], 'pid': p, 'event': P.loc[p, f'AE_Description({i})'], 'intensity': str(P.loc[p, f'ae_intensity({i})'])}
                           for p in AO + UC for i in (1, 2, 3) if pd.notna(P.loc[p].get(f'AE_Description({i})')) and P.loc[p, f'AE_Description({i})'] != '{null}']
    R['nirmatrelvir_ritonavir'] = pax[pax.arm.isin(['Antioxidant', 'Usual Care'])].to_dict('records')

    # ---- clinical outcomes: primary models and sensitivity analyses
    covs = ('age', 'male', 'vacc2', 'comorb', 'pax')
    # per-protocol: antioxidant participants who completed the course and took the first dose <= 5 days after onset
    st = df[df.redcap_event_name == 'Study treatment'].set_index('participant_id')
    first_dose = (pd.to_datetime(st.start_treat_date.reindex(AO), errors='coerce') - pd.to_datetime(P.symp_onset_date[AO])).dt.days
    completed = pd.to_numeric(st.end_treat_yn.reindex(AO), errors='coerce') == 1
    pp_excl = [p for p in AO if not completed[p] or first_dose[p] > 5]
    R['onset_to_first_dose_days'] = {'median': first_dose.median(), 'q1': first_dose.quantile(.25), 'q3': first_dose.quantile(.75), 'n_over_5': int((first_dose > 5).sum())}
    R['per_protocol_excluded_antioxidant'] = len(pp_excl)
    models = {}
    for o in ['hosp', 'rec', 'hea', 'act', 'all', 'sus']:
        models[o] = {'primary': fit(A, o, covs, seed=11), 'no_covariates': fit(A, o, (), seed=12),
                     'prior_sd_2.5': fit(A, o, covs, sd_beta=2.5, seed=13), 'prior_sd_0.5': fit(A, o, covs, sd_beta=0.5, seed=14)}
        if o != 'hosp':
            Aw = A.copy()
            Aw.loc[Aw.arm.eq('Antioxidant') & Aw[o].isna(), o] = 0.0
            Aw.loc[Aw.arm.eq('Usual Care') & Aw[o].isna(), o] = 1.0
            models[o]['worst_case_missing'] = fit(A, o, covs, seed=15, data=Aw)
        models[o]['per_protocol'] = fit(A[~A.index.isin(pp_excl)], o, covs, seed=16)
        print(o, {k: round(v['OR_median'], 2) for k, v in models[o].items()}, flush=True)
    R['models'] = models

    # ---- dietary recalls (S1 Table nutrients) and supplements (S2 Table)
    fd_cols = [c for c in pd.read_excel(F_DIARY, nrows=0).columns if c not in ('First Name', 'Last Name')]
    nutrients = S1_NUTRIENTS or fd_cols
    diet = dietary(arm, nutrients)
    diet.to_csv(os.path.join(OUT_DIR, 'dietary_recalls_paired.csv'), index=False)
    supplements(arm).to_csv(os.path.join(OUT_DIR, 'supplement_intake.csv'))
    R['dietary_paired_n'] = {'antioxidant': int(diet.ao_n.iloc[0]), 'usual_care': int(diet.uc_n.iloc[0])}

    json.dump(R, open(os.path.join(OUT_DIR, 'results.json'), 'w'), indent=1, default=str)
    rows = []
    for o, m in models.items():
        for k, s in m.items():
            rows.append({'outcome': o, 'analysis': k, **s})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'clinical_models.csv'), index=False)
    print('done; results in', OUT_DIR)


if __name__ == '__main__':
    main()
