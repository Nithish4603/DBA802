from pathlib import Path
import json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.base import clone

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'energy_efficiency.csv'
OUT = ROOT / 'outputs'
FIG = OUT / 'figures'
MET = OUT / 'metrics'
PRED = OUT / 'predictions'
for d in (FIG, MET, PRED): d.mkdir(parents=True, exist_ok=True)
SEED = 42
np.random.seed(SEED)

COLS = {
    'X1':'relative_compactness','X2':'surface_area','X3':'wall_area','X4':'roof_area',
    'X5':'overall_height','X6':'orientation','X7':'glazing_area','X8':'glazing_distribution',
    'Y1':'heating_load','Y2':'cooling_load'
}
FEATURES = list(COLS.values())[:8]
TARGETS = ['heating_load','cooling_load']


def metrics_for(y_true, y_pred):
    rows=[]
    for i,t in enumerate(TARGETS):
        rows.append({
            'target':t,
            'mae':mean_absolute_error(y_true[:,i], y_pred[:,i]),
            'rmse':mean_squared_error(y_true[:,i], y_pred[:,i])**0.5,
            'r2':r2_score(y_true[:,i], y_pred[:,i])
        })
    return rows


def make_models():
    models = {
        'linear_regression': Pipeline([('scale',StandardScaler()),('model',LinearRegression())]),
        'random_forest': RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1, min_samples_leaf=1),
        'gradient_boosting': MultiOutputRegressor(GradientBoostingRegressor(random_state=SEED, n_estimators=300, learning_rate=0.04, max_depth=3)),
        'mlp_neural_network': Pipeline([('scale',StandardScaler()),('model',MLPRegressor(hidden_layer_sizes=(64,32), max_iter=1500, early_stopping=True, random_state=SEED))]),
    }
    if HAVE_XGB:
        models['xgboost'] = MultiOutputRegressor(XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.9,
            colsample_bytree=0.9, objective='reg:squarederror', random_state=SEED,
            n_jobs=-1
        ))
    return models


def conformal_quantile(abs_residuals, alpha=0.10):
    n=len(abs_residuals)
    level=np.ceil((n+1)*(1-alpha))/n
    level=min(1.0, level)
    return float(np.quantile(abs_residuals, level, method='higher'))


def main():
    df=pd.read_csv(DATA).rename(columns=COLS)
    if not set(FEATURES+TARGETS).issubset(df.columns):
        raise ValueError('Dataset columns are not recognized.')
    X=df[FEATURES].copy(); y=df[TARGETS].copy()

    # 60/20/20 train/calibration/test split
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.40, random_state=SEED)
    X_cal, X_test, y_cal, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)

    models=make_models(); all_rows=[]; fitted={}; predictions={}
    for name,model in models.items():
        t0=time.perf_counter(); model.fit(X_train,y_train); train_s=time.perf_counter()-t0
        t1=time.perf_counter(); yp=model.predict(X_test); infer_s=time.perf_counter()-t1
        fitted[name]=model; predictions[name]=yp
        for row in metrics_for(y_test.values, yp):
            row.update({'model':name,'training_seconds':train_s,'inference_seconds':infer_s})
            all_rows.append(row)
    metrics=pd.DataFrame(all_rows)
    metrics.to_csv(MET/'model_target_metrics.csv',index=False)
    summary=metrics.groupby('model').agg(mae_mean=('mae','mean'),rmse_mean=('rmse','mean'),r2_mean=('r2','mean'),training_seconds=('training_seconds','first'),inference_seconds=('inference_seconds','first')).reset_index().sort_values('rmse_mean')
    summary.to_csv(MET/'model_comparison.csv',index=False)
    best_name=summary.iloc[0]['model']; best=fitted[best_name]

    # Conformal uncertainty using calibration residuals
    cal_pred=np.asarray(best.predict(X_cal)); test_pred=np.asarray(best.predict(X_test))
    qs=[]; coverage_rows=[]
    pred_df=X_test.reset_index(drop=True).copy()
    for i,t in enumerate(TARGETS):
        resid=np.abs(y_cal.iloc[:,i].values-cal_pred[:,i])
        q=conformal_quantile(resid,0.10); qs.append(q)
        lo=test_pred[:,i]-q; hi=test_pred[:,i]+q
        actual=y_test.iloc[:,i].values
        coverage=((actual>=lo)&(actual<=hi)).mean()
        width=(hi-lo).mean()
        coverage_rows.append({'target':t,'nominal_coverage':0.90,'empirical_coverage':coverage,'half_width':q,'mean_interval_width':width})
        pred_df[f'actual_{t}']=actual
        pred_df[f'predicted_{t}']=test_pred[:,i]
        pred_df[f'lower90_{t}']=lo
        pred_df[f'upper90_{t}']=hi
    pd.DataFrame(coverage_rows).to_csv(MET/'conformal_uncertainty.csv',index=False)

    # Data-driven decision tiers from training predictions of total load
    train_pred=np.asarray(best.predict(X_train))
    total_train=train_pred.sum(axis=1)
    q25,q50,q75=np.quantile(total_train,[.25,.5,.75])
    total_test=test_pred.sum(axis=1)
    tier=np.select([total_test<q25,total_test<q50,total_test<q75],['Low','Moderate','High'],default='Critical')
    pred_df['predicted_total_load']=total_test
    pred_df['decision_tier']=tier
    pred_df.to_csv(PRED/'test_predictions_with_uncertainty.csv',index=False)
    pd.DataFrame([{'q25':q25,'q50':q50,'q75':q75}]).to_csv(MET/'decision_thresholds.csv',index=False)

    # Prescriptive analytics: search feasible orientation/glazing alternatives observed in the dataset.
    # Candidates are restricted to (glazing_area, glazing_distribution) pairs that actually occur
    # together in the training distribution -- glazing_distribution=0 only ever co-occurs with
    # glazing_area=0 (a "no windows" design), so the full cartesian product used previously asked
    # the model to extrapolate to combinations it never saw in training. The zero-glazing option
    # is also excluded outright: it is not a realistic recommendation for an occupied building
    # (no daylighting, fails code minimums), even though it appears in the raw dataset.
    valid_combos = df.loc[df.glazing_area > 0, ['orientation', 'glazing_area', 'glazing_distribution']].drop_duplicates()
    presc=[]
    # Keep structure fixed, vary facade/orientation choices. Penalty mildly discourages large glazing changes.
    for ridx,row in X_test.reset_index(drop=True).iterrows():
        candidates=[]
        for _, combo in valid_combos.iterrows():
            c=row.copy(); c['orientation']=combo.orientation; c['glazing_area']=combo.glazing_area; c['glazing_distribution']=combo.glazing_distribution
            candidates.append(c)
        C=pd.DataFrame(candidates)[FEATURES]
        cp=np.asarray(best.predict(C))
        total=cp.sum(axis=1)
        change_penalty=2.0*np.abs(C.glazing_area.values-row.glazing_area)
        objective=total+change_penalty
        j=int(np.argmin(objective))
        base=test_pred[ridx]
        rec=cp[j]
        presc.append({
            'test_row':ridx,
            'current_orientation':row.orientation,'recommended_orientation':C.iloc[j].orientation,
            'current_glazing_area':row.glazing_area,'recommended_glazing_area':C.iloc[j].glazing_area,
            'current_glazing_distribution':row.glazing_distribution,'recommended_glazing_distribution':C.iloc[j].glazing_distribution,
            'current_pred_heating':base[0],'recommended_pred_heating':rec[0],
            'current_pred_cooling':base[1],'recommended_pred_cooling':rec[1],
            'current_pred_total':base.sum(),'recommended_pred_total':rec.sum(),
            'estimated_reduction':base.sum()-rec.sum(),
            'estimated_reduction_pct':100*(base.sum()-rec.sum())/max(base.sum(),1e-9),
            'recommended_total_lower90':rec.sum()-(qs[0]+qs[1]),
            'recommended_total_upper90':rec.sum()+(qs[0]+qs[1]),
        })
    presc_df=pd.DataFrame(presc)
    presc_df.to_csv(PRED/'prescriptive_recommendations.csv',index=False)
    presc_df[['estimated_reduction','estimated_reduction_pct']].describe().to_csv(MET/'prescriptive_summary.csv')

    # Permutation importance for selected model (multi-output R2 scorer)
    pi=permutation_importance(best,X_test,y_test,n_repeats=20,random_state=SEED,scoring='r2')
    imp=pd.DataFrame({'feature':FEATURES,'importance_mean':pi.importances_mean,'importance_std':pi.importances_std}).sort_values('importance_mean',ascending=False)
    imp.to_csv(MET/'feature_importance.csv',index=False)

    # Charts
    plt.figure(figsize=(8,5));
    s=summary.sort_values('r2_mean')
    plt.barh(s.model,s.r2_mean); plt.xlabel('Mean R² across heating and cooling'); plt.title('Predictive Performance by Model'); plt.tight_layout(); plt.savefig(FIG/'model_comparison_r2.png',dpi=180); plt.close()

    plt.figure(figsize=(8,5));
    s=summary.sort_values('rmse_mean',ascending=False)
    plt.barh(s.model,s.rmse_mean); plt.xlabel('Mean RMSE'); plt.title('Prediction Error by Model'); plt.tight_layout(); plt.savefig(FIG/'model_comparison_rmse.png',dpi=180); plt.close()

    for i,t in enumerate(TARGETS):
        plt.figure(figsize=(6,6));
        plt.scatter(y_test.iloc[:,i],test_pred[:,i],alpha=.65)
        mn=min(y_test.iloc[:,i].min(),test_pred[:,i].min()); mx=max(y_test.iloc[:,i].max(),test_pred[:,i].max())
        plt.plot([mn,mx],[mn,mx],linestyle='--'); plt.xlabel(f'Actual {t.replace("_"," ")}'); plt.ylabel(f'Predicted {t.replace("_"," ")}'); plt.title(f'Actual vs Predicted: {t.replace("_"," ").title()}'); plt.tight_layout(); plt.savefig(FIG/f'actual_vs_predicted_{t}.png',dpi=180); plt.close()

    # interval plot first 40 sorted by actual heating
    sample=pred_df.sort_values('actual_heating_load').head(40).reset_index(drop=True)
    x=np.arange(len(sample))
    plt.figure(figsize=(10,5));
    plt.errorbar(x,sample.predicted_heating_load,yerr=[sample.predicted_heating_load-sample.lower90_heating_load,sample.upper90_heating_load-sample.predicted_heating_load],fmt='o',capsize=2,label='Prediction ± 90% interval')
    plt.scatter(x,sample.actual_heating_load,marker='x',label='Actual')
    plt.xlabel('Selected test cases'); plt.ylabel('Heating load'); plt.title('Conformal Uncertainty Intervals (90%)'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'uncertainty_intervals_heating.png',dpi=180); plt.close()

    plt.figure(figsize=(8,5));
    ii=imp.sort_values('importance_mean')
    plt.barh(ii.feature,ii.importance_mean,xerr=ii.importance_std); plt.xlabel('Permutation importance (R² decrease)'); plt.title(f'Feature Importance: {best_name}'); plt.tight_layout(); plt.savefig(FIG/'feature_importance.png',dpi=180); plt.close()

    counts=pd.Series(tier).value_counts().reindex(['Low','Moderate','High','Critical'],fill_value=0)
    plt.figure(figsize=(7,5)); plt.bar(counts.index,counts.values); plt.ylabel('Test records'); plt.title('Decision Tier Distribution'); plt.tight_layout(); plt.savefig(FIG/'decision_tier_distribution.png',dpi=180); plt.close()

    vals=presc_df.estimated_reduction_pct.clip(lower=0)
    plt.figure(figsize=(8,5)); plt.hist(vals,bins=20); plt.xlabel('Estimated total-load reduction (%)'); plt.ylabel('Test designs'); plt.title('Prescriptive Improvement Distribution'); plt.tight_layout(); plt.savefig(FIG/'prescriptive_improvement.png',dpi=180); plt.close()

    run_info={
        'dataset_rows':len(df),'features':FEATURES,'targets':TARGETS,
        'train_rows':len(X_train),'calibration_rows':len(X_cal),'test_rows':len(X_test),
        'best_model':best_name,'random_seed':SEED,'xgboost_available':HAVE_XGB,
        'prescriptive_search':'orientation, glazing area, and glazing distribution varied over values observed in the dataset; structural geometry held fixed',
        'uncertainty_method':'split conformal prediction, 90% marginal intervals per target'
    }
    (MET/'run_info.json').write_text(json.dumps(run_info,indent=2))
    print('\nExperiment complete')
    print(summary.to_string(index=False))
    print('\nBest model:',best_name)
    print('\nUncertainty summary:')
    print(pd.DataFrame(coverage_rows).to_string(index=False))
    print('\nMedian estimated prescriptive reduction (%):',round(float(presc_df.estimated_reduction_pct.median()),2))
    print('Outputs:',OUT)

if __name__=='__main__': main()
