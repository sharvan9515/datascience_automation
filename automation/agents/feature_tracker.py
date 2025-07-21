import json
from typing import Dict, List, Set, Optional
from automation.pipeline_state import PipelineState
from .base import BaseAgent


class FeatureTracker:
    """Tracks feature history and provides intelligent feedback."""
    
    def __init__(self):
        self.feature_history: Dict[str, Dict] = {}
        self.rejected_features: Set[str] = set()
        self.accepted_features: Set[str] = set()
        self.feature_performance: Dict[str, float] = {}
        self.feature_reasons: Dict[str, str] = {}
        
    def track_feature_proposal(self, feature_name: str, formula: str, rationale: str) -> bool:
        """Track a new feature proposal and return whether it should be allowed."""
        if feature_name in self.rejected_features:
            return False  # Block previously rejected features
            
        if feature_name not in self.feature_history:
            self.feature_history[feature_name] = {
                'formula': formula,
                'rationale': rationale,
                'proposal_count': 0,
                'implementation_count': 0,
                'rejection_count': 0,
                'best_score': None,
                'rejection_reasons': []
            }
        
        self.feature_history[feature_name]['proposal_count'] += 1
        return True
    
    def track_feature_implementation(self, feature_name: str, score: float, accepted: bool, 
                                   rejection_reason: Optional[str] = None):
        """Track feature implementation result."""
        if feature_name not in self.feature_history:
            return
            
        self.feature_history[feature_name]['implementation_count'] += 1
        
        if accepted:
            self.accepted_features.add(feature_name)
            if self.feature_history[feature_name]['best_score'] is None or score > self.feature_history[feature_name]['best_score']:
                self.feature_history[feature_name]['best_score'] = score
        else:
            self.rejected_features.add(feature_name)
            self.feature_history[feature_name]['rejection_count'] += 1
            if rejection_reason:
                self.feature_history[feature_name]['rejection_reasons'].append(rejection_reason)
            self.feature_performance[feature_name] = score
            self.feature_reasons[feature_name] = rejection_reason or "Unknown"
    
    def get_rejection_summary(self) -> str:
        """Get a summary of rejected features for LLM feedback."""
        if not self.rejected_features:
            return ""
            
        summary = "\n\nBLOCKED FEATURES (DO NOT PROPOSE THESE AGAIN):\n"
        for feature_name in self.rejected_features:
            history = self.feature_history.get(feature_name, {})
            rejection_count = history.get('rejection_count', 0)
            best_score = history.get('best_score', 'N/A')
            reasons = history.get('rejection_reasons', [])
            
            summary += f"- {feature_name}: Rejected {rejection_count} times"
            if best_score != 'N/A':
                summary += f", Best score: {best_score:.4f}"
            if reasons:
                summary += f", Reasons: {', '.join(reasons[-2:])}"  # Last 2 reasons
            summary += "\n"
        
        return summary
    
    def get_successful_patterns(self) -> str:
        """Get patterns from successful features for guidance."""
        if not self.accepted_features:
            return ""
            
        summary = "\n\nSUCCESSFUL FEATURE PATTERNS:\n"
        for feature_name in self.accepted_features:
            history = self.feature_history.get(feature_name, {})
            formula = history.get('formula', 'Unknown')
            rationale = history.get('rationale', 'Unknown')
            best_score = history.get('best_score', 'N/A')
            
            summary += f"- {feature_name} = {formula} (Score: {best_score})\n"
            summary += f"  Rationale: {rationale}\n"
        
        return summary
    
    def get_feature_suggestions(self, existing_features: List[str]) -> str:
        """Get intelligent suggestions based on feature history."""
        suggestions = "\n\nFEATURE SUGGESTIONS:\n"
        
        # Analyze patterns in successful features
        successful_patterns = []
        for feature_name in self.accepted_features:
            history = self.feature_history.get(feature_name, {})
            formula = history.get('formula', '')
            if '+' in formula:
                successful_patterns.append("combining features with addition")
            elif '*' in formula:
                successful_patterns.append("combining features with multiplication")
            elif '**' in formula:
                successful_patterns.append("polynomial features")
        
        if successful_patterns:
            suggestions += f"- Successful patterns: {', '.join(set(successful_patterns))}\n"
        
        # Analyze rejection patterns
        rejection_patterns = []
        for feature_name in self.rejected_features:
            history = self.feature_history.get(feature_name, {})
            formula = history.get('formula', '')
            reasons = history.get('rejection_reasons', [])
            
            if any('correlation' in reason.lower() for reason in reasons):
                rejection_patterns.append("high correlation with existing features")
            if any('performance' in reason.lower() for reason in reasons):
                rejection_patterns.append("performance degradation")
            if any('variance' in reason.lower() for reason in reasons):
                rejection_patterns.append("high variance/noise")
        
        if rejection_patterns:
            suggestions += f"- Avoid: {', '.join(set(rejection_patterns))}\n"
        
        return suggestions
    
    def should_skip_feature(self, feature_name: str) -> bool:
        """Check if a feature should be skipped based on history."""
        return feature_name in self.rejected_features
    
    def get_feature_stats(self) -> Dict:
        """Get statistics about feature tracking."""
        total_proposed = len(self.feature_history)
        total_rejected = len(self.rejected_features)
        total_accepted = len(self.accepted_features)
        
        return {
            'total_proposed': total_proposed,
            'total_rejected': total_rejected,
            'total_accepted': total_accepted,
            'acceptance_rate': total_accepted / total_proposed if total_proposed > 0 else 0,
            'rejection_rate': total_rejected / total_proposed if total_proposed > 0 else 0
        }


class Agent(BaseAgent):
    """Feature tracking agent that maintains feature history and provides intelligent feedback."""
    
    def __init__(self):
        self.tracker = FeatureTracker()
    
    def run(self, state: PipelineState) -> PipelineState:
        state.append_log("FeatureTracker: analyzing feature history")
        
        # Initialize tracker if not exists
        if not hasattr(state, 'feature_tracker_instance') or state.feature_tracker_instance is None:
            state.feature_tracker_instance = self.tracker
        
        # Update tracker with recent snippet history (after feature implementation)
        if hasattr(state, 'snippet_history') and state.snippet_history:
            for entry in state.snippet_history[-20:]:  # Last 20 entries
                if entry.get('stage') == 'feature_implementation':
                    feature_name = entry.get('snippet', '').split('=')[0].strip() if '=' in entry.get('snippet', '') else 'unknown'
                    score = entry.get('score', 0.0)
                    accepted = entry.get('accepted', False)
                    strategy_type = entry.get('strategy_type', None)
                    rejection_reason = None
                    if not accepted and hasattr(state, 'log'):
                        recent_logs = state.log[-10:]
                        for log in recent_logs:
                            if 'correlation with target:' in log and feature_name in log:
                                rejection_reason = "low correlation with target"
                                break
                            elif 'detected high correlations:' in log and feature_name in log:
                                rejection_reason = "high correlation with existing features"
                                break
                            elif 'performance decrease' in log and feature_name in log:
                                rejection_reason = "performance degradation"
                                break
                    state.feature_tracker_instance.track_feature_implementation(
                        feature_name, score, accepted, rejection_reason
                    )
                    # Track strategy type success
                    if not hasattr(state.feature_tracker_instance, 'strategy_stats'):
                        state.feature_tracker_instance.strategy_stats = {}
                    if strategy_type:
                        stats = state.feature_tracker_instance.strategy_stats.setdefault(strategy_type, {'accepted': 0, 'rejected': 0})
                        if accepted:
                            stats['accepted'] += 1
                        else:
                            stats['rejected'] += 1
        
        # Store tracking information in state for other agents to use
        state.feature_tracking_summary = {
            'rejection_summary': state.feature_tracker_instance.get_rejection_summary(),
            'successful_patterns': state.feature_tracker_instance.get_successful_patterns(),
            'feature_suggestions': state.feature_tracker_instance.get_feature_suggestions(
                [c for c in state.df.columns if c != state.target]
            ),
            'stats': state.feature_tracker_instance.get_feature_stats(),
            'strategy_stats': getattr(state.feature_tracker_instance, 'strategy_stats', {})
        }
        
        # Log tracking summary
        stats = state.feature_tracker_instance.get_feature_stats()
        state.append_log(f"FeatureTracker: {stats['total_proposed']} features tracked, "
                        f"{stats['total_accepted']} accepted, {stats['total_rejected']} rejected")
        
        # Log strategy success rates
        strategy_stats = getattr(state.feature_tracker_instance, 'strategy_stats', {})
        if strategy_stats:
            for strat, s in strategy_stats.items():
                state.append_log(f"FeatureTracker: Strategy '{strat}' - accepted: {s['accepted']}, rejected: {s['rejected']}")
        
        if stats['total_rejected'] > 0:
            state.append_log(f"FeatureTracker: {len(state.feature_tracker_instance.rejected_features)} features blocked from re-proposal")
        
        return state


# Backwards compatible function API
def run(state: PipelineState) -> PipelineState:
    return Agent().run(state) 