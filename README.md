# 🏎️ EcoDrive Simulator

**AI-Powered Driving Optimization for Smart Mobility**

A comprehensive web application that demonstrates how reinforcement learning can optimize driving performance while minimizing environmental impact. Compare AI vs Human driving patterns and discover sustainable mobility solutions.

## ✨ Features

### 🎯 Core Functionality
- **Interactive Simulation Setup**: Choose from multiple tracks, driving modes, and conditions
- **Real-time Dashboard**: Live visualization of AI vs Human performance
- **Comprehensive Results**: Detailed comparison and analysis of driving efficiency
- **Educational Insights**: Learn about reinforcement learning and eco-driving principles
- **Admin Panel**: Monitor simulations, manage AI models, and analyze system performance

### 🎨 Design & UX
- **Futuristic Dark Theme**: Inspired by F1 telemetry dashboards
- **Energy Gradient Accents**: Green ↔ Orange ↔ Red reflecting eco vs aggressive driving
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Smooth Animations**: Framer Motion powered transitions and interactions

### 🧠 AI & Technology
- **Multiple RL Algorithms**: PPO, SAC, and TD3 model support
- **Real-time Telemetry**: Live data visualization and analysis
- **Performance Metrics**: Lap times, energy consumption, CO₂ emissions
- **Model Insights**: AI decision-making explanations and heatmaps

## 🚀 Quick Start

### Prerequisites
- Node.js (v16 or higher)
- Python (v3.8 or higher)
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EcoDrive-Simulator
   ```

2. **Install frontend dependencies**
   ```bash
   npm install
   ```

3. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Start the backend server**
   ```bash
   cd backend
   python app.py
   ```

5. **Start the frontend development server**
   ```bash
   npm start
   ```

6. **Open your browser**
   Navigate to `http://localhost:3000`

##  Project Structure

```
EcoDrive-Simulator/
├── public/                 # Static assets
├── src/
│   ├── components/         # Reusable React components
│   │   ├── Navigation.js   # Main navigation component
│   │   └── Navigation.css
│   ├── pages/             # Page components
│   │   ├── LandingPage.js # Home/landing page
│   │   ├── SimulationSetup.js # Simulation configuration
│   │   ├── SimulationDashboard.js # Live simulation view
│   │   ├── ResultsComparison.js # Results and analysis
│   │   ├── EducationalInsights.js # Learning content
│   │   └── AdminPanel.js  # Admin interface
│   ├── App.js             # Main app component
│   ├── App.css            # Global styles
│   ├── index.js           # App entry point
│   └── index.css          # Base styles
├── backend/
│   ├── app.py             # Flask API server
│   └── requirements.txt   # Python dependencies
└── README.md
```

## 🎮 Usage Guide

### 1. Landing Page
- Explore the platform introduction
- View key statistics and features
- Navigate to different sections

### 2. Simulation Setup
- **Track Selection**: Choose from Silverstone, Monza, Yas Marina, Spa, or Monaco
- **Driving Mode**: Select Eco, Balanced, or Aggressive driving styles
- **Conditions**: Configure weather and surface conditions
- **Advanced Settings**: Choose AI model version (PPO, SAC, TD3)

### 3. Simulation Dashboard
- **Live Visualization**: Track map with AI and Human racing lines
- **Real-time Metrics**: Lap times, energy consumption, CO₂ savings
- **Interactive Charts**: Speed, throttle, and energy consumption graphs
- **Model Insights**: AI decision-making explanations

### 4. Results & Comparison
- **Performance Summary**: Winner determination and key improvements
- **Detailed Analysis**: Comprehensive comparison charts and metrics
- **Key Insights**: Environmental impact and efficiency gains
- **Report Download**: Export results for further analysis

### 5. Educational Insights
- **Reinforcement Learning**: Learn the fundamentals of RL
- **AI Driving**: Understand how AI optimizes driving patterns
- **Eco-driving Principles**: Apply AI techniques to real-world driving
- **Interactive Learning**: Hands-on exploration of concepts

### 6. Admin Panel
- **System Overview**: Monitor total simulations and active users
- **Simulation Logs**: View detailed logs of all simulation runs
- **Model Management**: Monitor AI model performance and trigger retraining
- **System Health**: Track system resources and data processing

## 🛠️ Technology Stack

### Frontend
- **React 18**: Modern React with hooks and functional components
- **React Router**: Client-side routing and navigation
- **Framer Motion**: Smooth animations and transitions
- **Recharts**: Interactive charts and data visualization
- **Lucide React**: Modern icon library
- **CSS3**: Custom styling with futuristic design

### Backend
- **Flask**: Lightweight Python web framework
- **Flask-CORS**: Cross-origin resource sharing support
- **JSON**: Data serialization and API responses

### Development Tools
- **Create React App**: Development environment and build tools
- **npm**: Package management
- **Git**: Version control

## 🎨 Design System

### Color Palette
- **Primary Green**: `#00ff88` - Eco-friendly, AI optimization
- **Secondary Orange**: `#ff6b35` - Human performance, balanced
- **Accent Red**: `#ff1744` - Aggressive mode, alerts
- **Background**: `#0a0a0a` to `#1a1a1a` - Dark gradient
- **Text**: `#ffffff` with opacity variations

### Typography
- **Primary Font**: Montserrat - Clean, modern sans-serif
- **Secondary Font**: Poppins - Friendly, readable alternative

### Components
- **Glass Effect**: `backdrop-filter: blur(10px)` with transparency
- **Gradient Borders**: CSS gradients for dynamic borders
- **Hover Animations**: Scale, translate, and glow effects
- **Loading States**: Spinners and skeleton screens

## 🔧 API Endpoints

### Simulation Management
- `POST /api/simulation/start` - Start new simulation
- `POST /api/simulation/<id>/update` - Update simulation progress
- `POST /api/simulation/<id>/complete` - Complete simulation
- `GET /api/simulation/logs` - Get simulation logs

### System Management
- `GET /api/system/stats` - Get system statistics
- `GET /api/models/performance` - Get AI model performance
- `POST /api/models/<name>/retrain` - Trigger model retraining

### Data & Content
- `GET /api/tracks` - Get available tracks
- `POST /api/telemetry/generate` - Generate mock telemetry data
- `GET /api/insights/rl` - Get educational content

## 🚀 Deployment

### Frontend Deployment
```bash
npm run build
# Deploy the 'build' folder to your hosting service
```

### Backend Deployment
```bash
# Deploy to cloud platform (Heroku, AWS, etc.)
# Ensure Python environment and dependencies are installed
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **International Energy Agency (IEA)** - Sustainability data and insights
- **FIA Sustainability** - Motorsport sustainability initiatives
- **Reinforcement Learning Community** - Research and algorithm development
- **Open Source Community** - Libraries and tools that made this possible

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation and FAQ

---

**Built with ❤️ for sustainable mobility and AI education**
