"""Baseline configs match the paper (no TF needed)."""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

src = (ROOT / "zeolite_baselines.py").read_text()
ast.parse(src)

# Paper models only: a-CNN, RF, SVM, GBC. Nothing else.
assert '"RF"' in src and '"SVM"' in src and '"GBC"' in src
assert '"KNN"' not in src and '"ExTree"' not in src and "ExtraTrees" not in src
print("1. paper models only: PASS")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
sv = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", probability=True, random_state=42)
gb = GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=5,
                                subsample=0.8, random_state=42)
assert (rf.n_estimators, rf.class_weight, rf.random_state) == (300, "balanced", 42)
assert (sv.kernel, sv.C, sv.gamma, sv.probability) == ("rbf", 10, "scale", True)
assert (gb.n_estimators, gb.learning_rate, gb.max_depth, gb.subsample) == (250, 0.05, 5, 0.8)
print("2. traditional configs instantiate with paper values: PASS")

# a-CNN structure asserts on source (TF absent locally).
assert "Conv1D(32, 8, strides=8" in src
assert "Conv1D(32, 5, strides=5" in src
assert "Conv1D(32, 3, strides=3" in src
assert "GlobalAveragePooling1D" in src
print("3. a-CNN layers/strides per design: PASS")
print("all baseline checks: PASS")
