import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from tqdm.notebook import tqdm
from sklearn.svm import SVC


def do_cv_svm(X, y, cv_splits, params_cv_folds, params={'C' : [1, 10, 100, 1000], 'gamma' : ['auto', 'scale']}, n_jobs=1):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    acuracias = []
    best_params = {}
    fold = 1
    
    pgb = tqdm(total=cv_splits, desc='Folds avaliados')
    
    for treino_idx, teste_idx in skf.split(X, y):

        X_treino = X[treino_idx]
        y_treino = y[treino_idx]

        X_teste = X[teste_idx]
        y_teste = y[teste_idx]

        X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, stratify=y_treino, test_size=0.2, random_state=1)

        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)
        X_val = ss.transform(X_val)
        
        score = 'f1' if len(set(y_treino)) < 3 else 'f1_weighted'
        
        grid = GridSearchCV(SVC(), param_grid=params, cv=StratifiedKFold(n_splits=params_cv_folds), n_jobs=n_jobs, scoring=score)   
        grid.fit(np.vstack((X_treino, X_val)), [*y_treino, *y_val])
        melhor_val = grid.best_score_    
        melhor_comb = grid.best_params_
        
        best_params[melhor_val] = (melhor_comb, grid)
        
        pred = grid.predict(X_teste)

        # acuracias.append(f1_score(y_teste, pred, average='weighted'))
        acuracias.append(classification_report(y_teste, pred, output_dict=True))

        fold+=1
        pgb.update(1)
        
    pgb.close()
    
    return acuracias, best_params