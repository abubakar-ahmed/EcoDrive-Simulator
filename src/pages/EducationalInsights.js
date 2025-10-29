import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { 
  ArrowLeft, 
  BookOpen, 
  Brain, 
  Zap, 
  Leaf, 
  Target, 
  TrendingUp, 
  Play, 
  ChevronDown, 
  ChevronUp,
  Award,
  Clock,
  Users,
  Globe,
  Car,
  Gauge,
  Lightbulb,
  CheckCircle,
  ExternalLink,
  Download,
  Star
} from 'lucide-react';
import './EducationalInsights.css';

const EducationalInsights = () => {
  const [expandedSection, setExpandedSection] = useState(null);
  const [activeVideo, setActiveVideo] = useState(null);

  const insights = [
    {
      id: 'rl-fundamentals',
      icon: Brain,
      title: 'Reinforcement Learning Fundamentals',
      subtitle: 'Understanding the basics of AI learning',
      content: 'Reinforcement Learning (RL) is a type of machine learning where an AI agent learns to make decisions by interacting with an environment and receiving rewards or penalties for its actions. Unlike supervised learning, RL doesn\'t require labeled training data - instead, it learns through trial and error, gradually improving its performance over time.',
      detailedContent: `
        <h4>Key Components:</h4>
        <ul>
          <li><strong>Agent:</strong> The AI system making decisions</li>
          <li><strong>Environment:</strong> The world the agent interacts with</li>
          <li><strong>Actions:</strong> What the agent can do</li>
          <li><strong>Rewards:</strong> Feedback that guides learning</li>
          <li><strong>Policy:</strong> The strategy the agent uses to choose actions</li>
        </ul>
        
        <h4>How RL Works in Racing:</h4>
        <p>In our simulator, the AI agent learns to drive by:</p>
        <ol>
          <li>Observing track conditions, speed, and position</li>
          <li>Taking actions (accelerate, brake, steer)</li>
          <li>Receiving rewards for good driving (speed, efficiency)</li>
          <li>Learning from mistakes and improving over time</li>
        </ol>
      `,
      benefits: ['Self-learning capability', 'Adapts to new situations', 'Optimizes for long-term rewards'],
      videoId: '2pWv7GOvuf0', // Reinforcement Learning Explained
      duration: '8:40',
      difficulty: 'Beginner',
      category: 'Fundamentals'
    },
    {
      id: 'ppo-algorithm',
      icon: Zap,
      title: 'PPO Algorithm Deep Dive',
      subtitle: 'Proximal Policy Optimization explained',
      content: 'Proximal Policy Optimization (PPO) is a policy gradient method that uses a clipped objective function to prevent large policy updates, making training more stable. It\'s particularly effective for continuous control tasks like autonomous driving, where smooth and consistent actions are crucial.',
      detailedContent: `
        <h4>PPO Advantages:</h4>
        <ul>
          <li><strong>Stability:</strong> Clipped objective prevents destructive updates</li>
          <li><strong>Sample Efficiency:</strong> Uses data more effectively than older methods</li>
          <li><strong>Continuous Control:</strong> Perfect for driving tasks</li>
          <li><strong>Robustness:</strong> Works well across different environments</li>
        </ul>
        
        <h4>PPO in Our Simulator:</h4>
        <p>Our PPO model learns to:</p>
        <ul>
          <li>Optimize throttle and steering inputs</li>
          <li>Balance speed with energy efficiency</li>
          <li>Adapt to different track layouts</li>
          <li>Minimize lap times while conserving energy</li>
        </ul>
      `,
      benefits: ['Stable training', 'Sample efficient', 'Works well with continuous actions'],
      videoId: '5P7I-xPq8u8', // PPO Algorithm Explained
      duration: '12:15',
      difficulty: 'Intermediate',
      category: 'Algorithms'
    },
    {
      id: 'sac-algorithm',
      icon: Target,
      title: 'SAC Algorithm Explained',
      subtitle: 'Soft Actor-Critic for exploration',
      content: 'Soft Actor-Critic (SAC) is an off-policy algorithm that maximizes both expected reward and entropy, encouraging exploration while maintaining good performance. This makes it excellent for discovering novel eco-driving strategies that balance efficiency with performance.',
      detailedContent: `
        <h4>SAC Key Features:</h4>
        <ul>
          <li><strong>Entropy Regularization:</strong> Encourages exploration</li>
          <li><strong>Off-Policy Learning:</strong> Can learn from past experiences</li>
          <li><strong>Continuous Actions:</strong> Handles smooth control inputs</li>
          <li><strong>Sample Efficiency:</strong> Good data utilization</li>
        </ul>
        
        <h4>Why SAC for Eco-Driving:</h4>
        <p>SAC excels at finding creative solutions:</p>
        <ul>
          <li>Discovers unconventional but efficient racing lines</li>
          <li>Explores different energy management strategies</li>
          <li>Balances exploration with exploitation</li>
          <li>Adapts to changing track conditions</li>
        </ul>
      `,
      benefits: ['Encourages exploration', 'Handles continuous actions', 'Balances exploration and exploitation'],
      videoId: 'bRfUxQs7xJI', // SAC Algorithm Tutorial
      duration: '15:30',
      difficulty: 'Intermediate',
      category: 'Algorithms'
    },
    {
      id: 'eco-driving',
      icon: Leaf,
      title: 'Eco-Driving Principles',
      subtitle: 'Sustainable racing strategies',
      content: 'AI-powered eco-driving can reduce fuel consumption by 10-20% while maintaining competitive lap times, contributing to sustainable mobility. The AI learns optimal acceleration patterns, braking strategies, and energy management techniques that human drivers might not naturally adopt.',
      detailedContent: `
        <h4>Eco-Driving Techniques:</h4>
        <ul>
          <li><strong>Smooth Acceleration:</strong> Gradual speed increases</li>
          <li><strong>Anticipatory Braking:</strong> Early, gentle braking</li>
          <li><strong>Optimal Racing Lines:</strong> Minimize distance and energy</li>
          <li><strong>Energy Management:</strong> Strategic power usage</li>
        </ul>
        
        <h4>Environmental Impact:</h4>
        <p>Eco-driving benefits:</p>
        <ul>
          <li>Reduces CO₂ emissions by 15-25%</li>
          <li>Decreases fuel consumption significantly</li>
          <li>Extends vehicle lifespan</li>
          <li>Promotes sustainable racing practices</li>
        </ul>
      `,
      benefits: ['Reduced fuel consumption', 'Lower CO₂ emissions', 'Maintained performance levels'],
      videoId: 'Y4M9z4tUMeU', // Eco-Driving Techniques
      duration: '6:45',
      difficulty: 'Beginner',
      category: 'Sustainability'
    },
    {
      id: 'performance-optimization',
      icon: TrendingUp,
      title: 'Performance Optimization',
      subtitle: 'Continuous learning and adaptation',
      content: 'Our AI models continuously learn and adapt to different track conditions, weather patterns, and vehicle characteristics. This adaptive learning process ensures optimal performance across various scenarios, making each simulation more efficient than the last.',
      detailedContent: `
        <h4>Adaptive Learning Process:</h4>
        <ul>
          <li><strong>Track Adaptation:</strong> Learns each circuit's unique characteristics</li>
          <li><strong>Weather Response:</strong> Adjusts strategy for conditions</li>
          <li><strong>Vehicle Dynamics:</strong> Optimizes for specific car parameters</li>
          <li><strong>Continuous Improvement:</strong> Gets better with each lap</li>
        </ul>
        
        <h4>Optimization Metrics:</h4>
        <ul>
          <li>Lap time minimization</li>
          <li>Energy efficiency maximization</li>
          <li>Consistency across different conditions</li>
          <li>Adaptability to new scenarios</li>
        </ul>
      `,
      benefits: ['Continuous improvement', 'Adapts to conditions', 'Personalized strategies'],
      videoId: 'aircAruvnKk', // AI Performance Optimization
      duration: '10:20',
      difficulty: 'Advanced',
      category: 'Optimization'
    }
  ];

  const learningPaths = [
    {
      title: 'Beginner Path',
      description: 'Start your RL journey',
      topics: ['RL Fundamentals', 'Eco-Driving Principles'],
      duration: '30 min',
      icon: BookOpen
    },
    {
      title: 'Intermediate Path',
      description: 'Dive into algorithms',
      topics: ['PPO Algorithm', 'SAC Algorithm'],
      duration: '45 min',
      icon: Brain
    },
    {
      title: 'Advanced Path',
      description: 'Master optimization',
      topics: ['Performance Optimization', 'Advanced Techniques'],
      duration: '60 min',
      icon: Award
    }
  ];

  const stats = [
    { icon: Users, value: '10,000+', label: 'Students Learned' },
    { icon: Clock, value: '2.5M', label: 'Hours Saved' },
    { icon: Globe, value: '50+', label: 'Countries Reached' },
    { icon: Star, value: '4.9/5', label: 'Average Rating' }
  ];

  const toggleSection = (sectionId) => {
    setExpandedSection(expandedSection === sectionId ? null : sectionId);
  };

  const openVideo = (videoId) => {
    setActiveVideo(videoId);
  };

  const closeVideo = () => {
    setActiveVideo(null);
  };

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'Beginner': return '#00ff88';
      case 'Intermediate': return '#ff6b35';
      case 'Advanced': return '#ff1744';
      default: return '#ffffff';
    }
  };

  return (
    <div className="educational-insights">
      <div className="container">
        {/* Hero Section */}
        <div className="hero-section">
          <div className="hero-content">
            <Link to="/" className="back-button">
              <ArrowLeft size={20} />
              Back to Home
            </Link>
            <h1 className="hero-title">
              <BookOpen size={32} />
              Educational Insights
            </h1>
            <p className="hero-description">
              Master reinforcement learning, AI algorithms, and eco-driving technologies through interactive content and expert videos
            </p>
          </div>
          
          <div className="hero-stats">
            {stats.map((stat, index) => (
              <div key={index} className="stat-item">
                <stat.icon size={20} />
                <div className="stat-value">{stat.value}</div>
                <div className="stat-label">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Learning Paths */}
        <div className="learning-paths-section">
          <h2>Choose Your Learning Path</h2>
          <div className="learning-paths-grid">
            {learningPaths.map((path, index) => (
              <div key={index} className="learning-path-card">
                <div className="path-icon">
                  <path.icon size={24} />
                </div>
                <h3>{path.title}</h3>
                <p>{path.description}</p>
                <div className="path-topics">
                  {path.topics.map((topic, idx) => (
                    <span key={idx} className="topic-tag">{topic}</span>
                  ))}
                </div>
                <div className="path-duration">
                  <Clock size={16} />
                  {path.duration}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="insights-content">
          {insights.map((insight, index) => (
            <div key={insight.id} className="insight-section">
              <div className="insight-card">
                <div className="insight-header">
                  <div className="insight-icon">
                    <insight.icon size={24} />
                  </div>
                  <div className="insight-title-section">
                    <h2>{insight.title}</h2>
                    <p className="insight-subtitle">{insight.subtitle}</p>
                    <div className="insight-meta">
                      <span className="difficulty-badge" style={{ backgroundColor: getDifficultyColor(insight.difficulty) }}>
                        {insight.difficulty}
                      </span>
                      <span className="category-badge">{insight.category}</span>
                      <span className="duration-badge">
                        <Clock size={14} />
                        {insight.duration}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="insight-content">
                  <p className="insight-summary">{insight.content}</p>
                  
                  {/* Video Section */}
                  <div className="video-section">
                    <div className="video-preview" onClick={() => openVideo(insight.videoId)}>
                      <div className="video-thumbnail">
                        <img 
                          src={`https://img.youtube.com/vi/${insight.videoId}/maxresdefault.jpg`}
                          alt={insight.title}
                          onError={(e) => {
                            e.target.src = `https://img.youtube.com/vi/${insight.videoId}/hqdefault.jpg`;
                          }}
                        />
                        <div className="play-button">
                          <Play size={32} />
                        </div>
                        <div className="video-duration">{insight.duration}</div>
                      </div>
                      <div className="video-info">
                        <h4>Watch: {insight.title}</h4>
                        <p>Expert explanation and visual demonstrations</p>
                      </div>
                    </div>
                  </div>

                  {/* Expandable Content */}
                  <div className="expandable-content">
                    <button 
                      className="expand-button"
                      onClick={() => toggleSection(insight.id)}
                    >
                      {expandedSection === insight.id ? (
                        <>
                          <ChevronUp size={20} />
                          Show Less
                        </>
                      ) : (
                        <>
                          <ChevronDown size={20} />
                          Learn More
                        </>
                      )}
                    </button>
                    
                    {expandedSection === insight.id && (
                      <div className="expanded-content">
                        <div 
                          className="detailed-content"
                          dangerouslySetInnerHTML={{ __html: insight.detailedContent }}
                        />
                        
                        <div className="benefits-list">
                          <h4>
                            <CheckCircle size={20} />
                            Key Benefits
                          </h4>
                          <ul>
                            {insight.benefits.map((benefit, idx) => (
                              <li key={idx}>
                                <CheckCircle size={16} />
                                {benefit}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Additional Resources */}
        <div className="resources-section">
          <h2>Additional Resources</h2>
          <div className="resources-grid">
            <div className="resource-card">
              <div className="resource-icon">
                <Download size={24} />
              </div>
              <h3>Download Study Guide</h3>
              <p>Comprehensive PDF guide covering all RL concepts</p>
              <button className="resource-button">
                <Download size={16} />
                Download PDF
              </button>
            </div>
            
            <div className="resource-card">
              <div className="resource-icon">
                <ExternalLink size={24} />
              </div>
              <h3>External Resources</h3>
              <p>Curated links to papers, tutorials, and tools</p>
              <button className="resource-button">
                <ExternalLink size={16} />
                View Links
              </button>
            </div>
            
            <div className="resource-card">
              <div className="resource-icon">
                <Lightbulb size={24} />
              </div>
              <h3>Practice Exercises</h3>
              <p>Hands-on coding exercises and challenges</p>
              <button className="resource-button">
                <Play size={16} />
                Start Practice
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Video Modal */}
      {activeVideo && (
        <div className="video-modal" onClick={closeVideo}>
          <div className="video-modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-video" onClick={closeVideo}>
              ×
            </button>
            <div className="video-container">
              <iframe
                width="100%"
                height="100%"
                src={`https://www.youtube.com/embed/${activeVideo}?autoplay=1&rel=0`}
                title="Educational Video"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              ></iframe>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EducationalInsights;