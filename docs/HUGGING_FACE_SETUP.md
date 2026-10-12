# Map prototype and Hugging Face inference

The completed design prototype is `frontend/prototypes/map-original.tsx`. It is separate from the active research map. It now exports a component, uses valid latitude/longitude fields, accepts real prediction points, supports point/weighted-density layers, shows empty/error states, and cleans up map resources. It uses the installed MapLibre package directly, avoiding the original missing react-map-gl dependency. The incomplete Deck.gl hexagon layer was not retained: summing or averaging conditional probabilities across arbitrary points would need a defined interpretation. No dummy San Francisco data is loaded.

## Use the prototype

From a component in frontend/src:

```tsx
import DeckGLMap from '../prototypes/map-original'
import type { CrashRiskPoint } from '../prototypes/map-original'

// Supply results returned by your Python inference layer.
const points: CrashRiskPoint[] = []
export default function MapDesign() {
  return <DeckGLMap points={points} height={600} />
}
```

Do not replace the current App unless you want to switch designs. Type-check the prototype using `npx tsc -p tsconfig.prototype.json` from frontend. Python Joblib cannot run directly inside React. A future API can call predict_records and return its JSON list to the points prop; no HTTP inference endpoint is created by this scaffold.

## Load the model

From the project root in a compatible Python environment:

```bash
python -m pip install -r requirements-huggingface.txt
export HF_MODEL_REPO="YOUR_USERNAME/YOUR_MODEL_REPO"
# Prefer the exact model repository commit for repeatable inference.
export HF_MODEL_REVISION="YOUR_FULL_COMMIT_HASH"
python -m src.models.hf_inference --input examples/severity-input.json
```

Set your actual repo ID and revision. The default artifact filename is `final_severity_logistic_2021_2024.joblib`, matching the saved model. If you upload it under `model.joblib`, pass `--filename model.joblib`. For private repositories, set HF_TOKEN in the Python/server environment; never expose it through frontend/VITE environment variables or commit it. No token is required for a public repository.

## Start the area-ranking service

The Spring chatbot can ask the Python inference service to rank the existing 1 km² Atlanta grid cells by predicted serious/fatal severity. Install the inference dependencies in the environment that starts the service:

```bash
conda activate /Users/alexanderajagba/Documents/atlanta-crash-risk-predictor/.conda
python -m pip install -r requirements-inference.txt
python -m src.inference_api --local-model models/final_severity_logistic_2021_2024.joblib
```

The service listens on `http://127.0.0.1:8001` by default. Set `INFERENCE_URL` when Spring Boot needs another address. The ranking endpoint accepts:

```http
POST /areas/rank
Content-Type: application/json

{"scenarioDatetime":"2024-06-03 08:00:00","topK":5}
```

Every area is scored using the same Atlanta-local scenario timestamp. The returned `riskProbability` is predicted serious/fatal severity conditional on a recorded crash; it is not the probability that a crash occurs in that area. The endpoint does not apply a hard severity label because a validated decision threshold is not available.

The loader follows the official [Hugging Face download API](https://huggingface.co/docs/huggingface_hub/guides/download): hf_hub_download resolves and caches the artifact, and Joblib loads the resulting local path. Repository ID, revision and filename are configurable. Model schema and class order are checked after loading. Only load a trusted Joblib artifact. The local `src.models.preprocess` module must remain available because the saved pipeline references its category-normalization function. Dependencies match the saved model's inference requirements.

Python usage:

```python
from src.models.hf_inference import load_model, make_features, predict_records

model = load_model(repo_id="YOUR_USERNAME/YOUR_MODEL_REPO", revision="YOUR_FULL_COMMIT_HASH")
records = [{
    "datetime": "2024-06-03 08:00:00",
    "latitude": 33.76,
    "longitude": -84.39,
    "geolocated_city": "Atlanta",
    "geolocated_county": "Fulton",
}]
X = make_features(records)
probabilities = model.predict_proba(X)[:, 1]
# Possible mechanically, but the default 0.5 cutoff is not validated:
# labels = model.predict(X)
points = predict_records(model, records)
```

Inputs must provide Atlanta local wall-clock time without a timezone offset, valid coordinates and nonempty city/county strings. Features reproduce the training definitions. In particular, Monday=0, rush hours=7–9 or 16–18 inclusive, night=20–23 or 0–5. Results describe serious/fatal severity conditional on a recorded crash. Coordinate range checks do not establish geographic support; retain the existing research-map evidence screen before treating scores as applicable to a location. Use the same time scenario across compared locations.

## Test without a Hugging Face repository

```bash
python -m src.models.hf_inference \
  --local-model models/final_severity_logistic_2021_2024.joblib \
  --input examples/severity-input.json
```

Three tests verify feature construction/invalid-input rejection, local model inference, and download-argument forwarding using a mocked Hugging Face download. Actual hosted download was not exercised because no repository was supplied. No model was uploaded or retrained. Existing research-map and model artifacts remain unchanged.
