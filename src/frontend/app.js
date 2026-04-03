document.addEventListener('DOMContentLoaded', () => {
    const smilesInput   = document.getElementById('smiles-input');
    const predictBtn    = document.getElementById('predict-btn');
    const btnText       = document.querySelector('.btn-text');
    const spinner       = document.getElementById('btn-spinner');
    const errorMsg      = document.getElementById('error-msg');
    const resultsPanel  = document.getElementById('results-panel');
    const molImage      = document.getElementById('mol-image');
    const resLabel      = document.getElementById('res-label');
    const resName       = document.getElementById('res-name');
    const resConfidence = document.getElementById('res-confidence');
    const resWeight     = document.getElementById('res-weight');
    const resSmiles     = document.getElementById('res-smiles');

    const API_URL = 'http://127.0.0.1:8000/predict';

    predictBtn.addEventListener('click', handleAnalyze);
    smilesInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') handleAnalyze();
    });

    async function handleAnalyze() {
        const smiles = smilesInput.value.trim();
        if (!smiles) { showError('Please enter a SMILES sequence.'); return; }

        hideError();
        setLoading(true);

        try {
            const res = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ smiles })
            });

            if (!res.ok) {
                const errBody = await res.json().catch(() => ({}));
                // FastAPI Pydantic validation errors: detail is an array of {loc, msg, type}
                // HTTP exceptions: detail is a plain string
                let message;
                if (Array.isArray(errBody.detail)) {
                    message = errBody.detail
                        .map(e => e.msg || JSON.stringify(e))
                        .join(' · ');
                } else if (typeof errBody.detail === 'string') {
                    message = errBody.detail;
                } else {
                    message = `Server error: ${res.status}`;
                }
                throw new Error(message);
            }

            displayResults(await res.json());

        } catch (err) {
            console.error(err);
            showError(err.message || 'Could not connect to the server. Make sure the API is running.');
        } finally {
            setLoading(false);
        }
    }

    function setLoading(on) {
        btnText.textContent = on ? 'Analyzing…' : 'Analyze';
        spinner.classList.toggle('loader-hidden', !on);
        smilesInput.disabled = on;
        predictBtn.disabled  = on;
    }

    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.style.display = 'block';
    }

    function hideError() {
        errorMsg.style.display = 'none';
    }

    function displayResults(data) {
        resultsPanel.style.display = 'block';

        molImage.src  = `data:image/png;base64,${data.molecule_structure}`;
        resSmiles.textContent = data.smiles;
        resName.textContent   = data['common-name'] || '—';

        resLabel.textContent = data.label;
        resLabel.className   = 'stat-val ' + (data.bbb_permeable ? 'status-pos' : 'status-neg');

        resConfidence.textContent = `${(data.confidence * 100).toFixed(2)} %`;
        resWeight.textContent     = `${data.molecular_weight.toFixed(2)} g/mol`;

        // smooth scroll so results are visible
        resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});
