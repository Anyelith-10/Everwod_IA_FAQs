import { useEffect, useMemo, useState } from "react";
import "./App.css";

function getInitials(name = "") {
  return name
    .split(" ")
    .slice(0, 2)
    .map((w) => w[0])
    .join("")
    .toUpperCase();
}

function getScoreClass(score) {
  const num = parseFloat(score);
  if (num >= 75) return "high";
  if (num >= 50) return "medium";
  return "low";
}

function FaqCard({ faq, onApprove, onReject }) {
  const [leaving, setLeaving] = useState(false);

  const handleAction = (action) => {
    setLeaving(true);
    setTimeout(() => action(faq.id), 280);
  };

  return (
    <div className={`card ${leaving ? "leaving" : ""}`}>
      <div className="card-header">
        <div className="company-info">
          <div className="avatar">{getInitials(faq.company_name)}</div>
          <span className="company-name">{faq.company_name}</span>
        </div>
        <span className={`score-pill ${getScoreClass(faq.cluster_score)}`}>
          Score {faq.cluster_score}
        </span>
      </div>

      <div className="card-body">
        <div className="qa-grid">
          <div>
            <div className="field-label">Pregunta</div>
            <p className="question-text">{faq.question}</p>
          </div>
          <div>
            <div className="field-label">Respuesta</div>
            <p className="answer-text">{faq.answer}</p>
          </div>
        </div>

        {faq.support_examples?.length > 0 && (
          <div className="examples-box">
            <div className="field-label" style={{ marginBottom: "8px" }}>
              Ejemplos de soporte
            </div>
            <ul className="examples-list">
              {faq.support_examples.map((ex, i) => (
                <li key={i} className="example-item">{ex}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="card-footer">
        <span className="footer-id">ID #{faq.id}</span>
        <div className="actions">
          <button className="btn-approve" onClick={() => handleAction(onApprove)}>
            ✓ Aprobar
          </button>
          <button className="btn-reject" onClick={() => handleAction(onReject)}>
            ✕ Rechazar
          </button>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [faqs, setFaqs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedCompany, setSelectedCompany] = useState("all");
  const [approved, setApproved] = useState(0);
  const [rejected, setRejected] = useState(0);

  const fetchFaqs = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8004/faq/pending");
      const data = await response.json();
      setFaqs(data);
    } catch (error) {
      console.error("Error loading FAQs:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFaqs();
  }, []);

  const generateFaqs = async () => {
    try {
      setLoading(true);
      await fetch("http://127.0.0.1:8003/suggest", { method: "POST" });
      await fetchFaqs();
    } catch (error) {
      console.error("Error generating FAQs:", error);
    } finally {
      setLoading(false);
    }
  };

  const approveFaq = async (id) => {
    try {
      await fetch(`http://127.0.0.1:8004/faq/${id}/approve`, { method: "POST" });
      setFaqs((prev) => prev.filter((faq) => faq.id !== id));
      setApproved((n) => n + 1);
    } catch (error) {
      console.error(error);
    }
  };

  const rejectFaq = async (id) => {
    try {
      await fetch(`http://127.0.0.1:8004/faq/${id}/reject`, { method: "POST" });
      setFaqs((prev) => prev.filter((faq) => faq.id !== id));
      setRejected((n) => n + 1);
    } catch (error) {
      console.error(error);
    }
  };

  const companies = useMemo(
    () => [...new Set(faqs.map((faq) => faq.company_name))],
    [faqs]
  );

  const filteredFaqs = useMemo(
    () =>
      selectedCompany === "all"
        ? faqs
        : faqs.filter((faq) => faq.company_name === selectedCompany),
    [faqs, selectedCompany]
  );

  if (loading) {
    return <div className="loading">Cargando FAQs...</div>;
  }

  return (
    <div className="page">
      {/* Header */}
      <div className="header">
        <div>
          <h1 className="title">FAQs Everwod</h1>
          <p className="subtitle">Revisión y aprobación de FAQs generadas por IA</p>
        </div>
        <div className="header-actions">
          <button className="generate-btn" onClick={generateFaqs}>
            Generar FAQs
          </button>
          <div className="counter-badge">
            <span className="counter-dot" />
            {filteredFaqs.length} pendientes
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-label">Pendientes</div>
          <div className="stat-value">{faqs.length}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Aprobadas</div>
          <div className="stat-value green">{approved}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Rechazadas</div>
          <div className="stat-value red">{rejected}</div>
        </div>
      </div>

      {/* Toolbar */}
      <div className="toolbar">
        <select
          className="company-select"
          value={selectedCompany}
          onChange={(e) => setSelectedCompany(e.target.value)}
        >
          <option value="all">Todas las empresas</option>
          {companies.map((company) => (
            <option key={company} value={company}>
              {company}
            </option>
          ))}
        </select>
      </div>

      {/* Cards */}
      <div className="cards-grid">
        {filteredFaqs.length === 0 ? (
          <div className="empty-state">No hay FAQs pendientes de revisión.</div>
        ) : (
          filteredFaqs.map((faq) => (
            <FaqCard
              key={faq.id}
              faq={faq}
              onApprove={approveFaq}
              onReject={rejectFaq}
            />
          ))
        )}
      </div>
    </div>
  );
}

export default App;
