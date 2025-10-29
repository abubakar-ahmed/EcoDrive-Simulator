import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Menu, X, Zap, Play, BarChart3, TrendingUp, BookOpen, Settings, Code } from 'lucide-react';
import './Navigation.css';

const Navigation = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Home', icon: Zap },
    { path: '/simulator', label: 'Simulator', icon: Play },
    { path: '/dashboard', label: 'Dashboard', icon: BarChart3 },
    { path: '/results', label: 'Results', icon: TrendingUp },
    { path: '/insights', label: 'Insights', icon: BookOpen },
    { path: '/developer', label: 'Developer', icon: Code },
    { path: '/admin', label: 'Admin', icon: Settings },
  ];

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <motion.nav 
      className="navbar"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="nav-container">
        <Link to="/" className="nav-brand">
          <div className="brand-icon">
            <Zap size={32} />
          </div>
          <span className="brand-text">EcoDrive Simulator</span>
        </Link>

        <div className={`nav-menu ${isMenuOpen ? 'nav-menu-active' : ''}`}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`nav-link ${isActive ? 'active' : ''}`}
                onClick={() => setIsMenuOpen(false)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </div>

        <button 
          className="nav-toggle"
          onClick={toggleMenu}
          aria-label="Toggle navigation menu"
        >
          {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>
    </motion.nav>
  );
};

export default Navigation;
