import json
from typing import Dict, List, Optional, Union
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from text_utils import sanitize_text

@dataclass
class QA:
    question: str
    answer: Optional[str]
    evidence: List[str]
    category: Optional[int] = None
    adversarial_answer: Optional[str] = None

    @property
    def final_answer(self) -> Optional[str]:
        """Get the appropriate answer based on category."""
        if self.category == 5:
            return self.adversarial_answer
        return self.answer

@dataclass
class Turn:
    speaker: str
    dia_id: str
    text: str

@dataclass
class Session:
    session_id: int
    date_time: str
    turns: List[Turn]

@dataclass
class Conversation:
    speaker_a: str
    speaker_b: str
    sessions: Dict[int, Session]

@dataclass
class EventSummary:
    events: Dict[str, Dict[str, List[str]]]  # session -> speaker -> events

@dataclass
class Observation:
    observations: Dict[str, Dict[str, List[List[str]]]]  # session -> speaker -> [observation, evidence]

@dataclass
class LoCoMoSample:
    """A single sample from the LoComo dataset"""
    sample_id: str
    qa: List[QA]
    conversation: Conversation
    event_summary: EventSummary
    observation: Observation
    session_summary: Dict[str, str]

def parse_session(session_data: List[dict], session_id: int, date_time: str) -> Session:
    """Parse a single session's data, including turns with images by using their captions."""
    turns = []
    for turn in session_data:
        # For turns with images, combine caption and text
        text = turn.get("text", "")
        if "img_url" in turn and "blip_caption" in turn:
            caption_text = f"[Image: {turn['blip_caption']}]"
            if text:
                text = f"{caption_text} {text}"
            else:
                text = caption_text
            
        turns.append(Turn(
            speaker=sanitize_text(turn["speaker"]),
            dia_id=sanitize_text(turn["dia_id"]),
            text=sanitize_text(text)
        ))
    return Session(session_id=session_id, date_time=sanitize_text(date_time), turns=turns)

def parse_conversation(conv_data: dict) -> Conversation:
    """Parse conversation data."""
    sessions = {}
    for key, value in conv_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_id = int(key.split("_")[1])
            date_time = conv_data.get(f"{key}_date_time")
            if date_time:
                session = parse_session(value, session_id, date_time)
                # Only add sessions that have turns after filtering
                if session.turns:
                    sessions[session_id] = session
    
    return Conversation(
        speaker_a=sanitize_text(conv_data["speaker_a"]),
        speaker_b=sanitize_text(conv_data["speaker_b"]),
        sessions=sessions
    )

def load_json_data(filepath: str) -> dict:
    """Load JSON data from file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(sanitize_text(f"Error loading JSON file: {e}"))
        return {}

def load_locomo_dataset(file_path: Union[str, Path]) -> List[LoCoMoSample]:
    """
    Load the LoComo dataset from a JSON file, including image-based content by using captions.
    
    Args:
        file_path: Path to the JSON file containing the dataset
        
    Returns:
        List of LoCoMoSample objects containing the parsed data
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
        
    if not file_path.exists():
        raise FileNotFoundError(sanitize_text(f"Dataset file not found at {file_path}"))
    
    print(sanitize_text(f"Loading dataset from {file_path}"))
    data = load_json_data(file_path)
    
    samples = []
    total_qa = 0
    total_image_qa = 0
    qa_counts_per_sample = []
    
    for sample_idx, sample in enumerate(data):
        try:
            # Parse QA data
            qa_list = []
            sample_qa_count = 0
            sample_image_qa_count = 0
            
            for qa_idx, qa in enumerate(sample["qa"]):
                try:
                    # Check if QA has image evidence
                    has_image_evidence = False
                    for evidence_id in qa.get("evidence", []):
                        if ":" not in evidence_id:
                            continue
                        turn_id = evidence_id.split(":")[1]
                        for session in sample["conversation"].values():
                            if isinstance(session, list):
                                for turn in session:
                                    if turn.get("dia_id", "").endswith(turn_id):
                                        if "img_url" in turn or "blip_caption" in turn:
                                            has_image_evidence = True
                                            break
                    
                    if has_image_evidence:
                        sample_image_qa_count += 1
                        
                    qa_obj = QA(
                        question=sanitize_text(qa["question"]),
                        answer=sanitize_text(qa.get("answer")),
                        evidence=[sanitize_text(e) for e in qa.get("evidence", [])],
                        category=qa.get("category"),
                        adversarial_answer=sanitize_text(qa.get("adversarial_answer"))
                    )
                    qa_list.append(qa_obj)
                    sample_qa_count += 1
                    
                except KeyError as e:
                    print(sanitize_text(f"Error in sample {sample_idx}, QA pair {qa_idx}:"))
                    print(sanitize_text(f"QA data: {qa}"))
                    raise e
                except Exception as e:
                    print(sanitize_text(f"Unexpected error in sample {sample_idx}, QA pair {qa_idx}:"))
                    print(sanitize_text(f"QA data: {qa}"))
                    raise e
            
            # Parse conversation
            conversation = parse_conversation(sample["conversation"])
            
            # Parse event summary
            event_summary = EventSummary(events={k: {sk: [sanitize_text(e) for e in ev] for sk, ev in v.items()} for k, v in sample["event_summary"].items()})
            
            # Parse observation
            observation = Observation(observations={k: {sk: [[sanitize_text(o) for o in obs] for obs in ob] for sk, ob in v.items()} for k, v in sample["observation"].items()})
            
            # Get session summary
            session_summary = {k: sanitize_text(v) for k, v in sample.get("session_summary", {}).items()}
            
            # Create sample object
            sample_obj = LoCoMoSample(
                sample_id=sanitize_text(str(sample_idx)),
                qa=qa_list,
                conversation=conversation,
                event_summary=event_summary,
                observation=observation,
                session_summary=session_summary
            )
            samples.append(sample_obj)
            
            total_qa += sample_qa_count
            total_image_qa += sample_image_qa_count
            qa_counts_per_sample.append(sample_qa_count)
            
            # Print statistics for this sample
            print(sanitize_text(f"\nSample {sample_idx}:"))
            print(sanitize_text(f"  Total QAs: {sample_qa_count}"))
            print(sanitize_text(f"  QAs with image evidence: {sample_image_qa_count}"))
            
        except Exception as e:
            print(sanitize_text(f"Error processing sample {sample_idx}:"))
            print(sanitize_text(str(e)))
            raise e
    
    # Print overall statistics
    print(sanitize_text("\nOverall Statistics:"))
    print(sanitize_text(f"Total QAs: {total_qa}"))
    print(sanitize_text(f"Total QAs with image evidence: {total_image_qa}"))
    print(sanitize_text(f"Average QAs per sample: {total_qa / len(samples):.2f}"))
    print(sanitize_text(f"Min QAs in a sample: {min(qa_counts_per_sample)}"))
    print(sanitize_text(f"Max QAs in a sample: {max(qa_counts_per_sample)}"))
    
    return samples

def get_dataset_statistics(samples: List[LoCoMoSample]) -> Dict:
    """
    Get basic statistics about the text-only dataset.
    
    Args:
        samples: List of LoCoMoSample objects
        
    Returns:
        Dictionary containing various statistics about the dataset
    """
    stats = {
        "num_samples": len(samples),
        "total_qa_pairs": sum(len(sample.qa) for sample in samples),
        "total_sessions": sum(len(sample.conversation.sessions) for sample in samples),
        "total_turns": sum(
            sum(len(session.turns) for session in sample.conversation.sessions.values())
            for sample in samples
        ),
        "qa_with_adversarial": sum(
            sum(1 for qa in sample.qa if qa.adversarial_answer is not None)
            for sample in samples
        )
    }
    return stats

if __name__ == "__main__":
    # Example usage
    dataset_path = Path(__file__).parent / "data" / "locomo10.json"
    try:
        print(sanitize_text(f"Loading dataset from: {dataset_path}"))
        samples = load_locomo_dataset(dataset_path)
        for sample_idx, sample in enumerate(samples):
            print(sanitize_text(f"\nSample {sample_idx}:"))
            for _,turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    print(turn)
                    break   
        # stats = get_dataset_statistics(samples)
        # print("\nDataset Statistics (Text-only content):")
        # for key, value in stats.items():
        #     print(f"{key}: {value}")
        # print(len(samples))
        # for sample in samples:
        #     print(sample)
        #     break
    except Exception as e:
        print(sanitize_text(f"Error loading dataset: {e}"))
        raise