import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import LandingPage from './pages/LandingPage';
import SimulationSetup from './pages/SimulationSetup';
import SimulationDashboard from './pages/SimulationDashboard';
import ResultsComparison from './pages/ResultsComparison';
import EducationalInsights from './pages/EducationalInsights';
import Developer from './pages/Developer';
import AdminPanel from './pages/AdminPanel';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Navigation />
        <main>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/simulator" element={<SimulationSetup />} />
            <Route path="/dashboard" element={<SimulationDashboard />} />
            <Route path="/results" element={<ResultsComparison />} />
            <Route path="/insights" element={<EducationalInsights />} />
            <Route path="/developer" element={<Developer />} />
            <Route path="/admin" element={<AdminPanel />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;