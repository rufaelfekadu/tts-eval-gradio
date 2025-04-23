import os
import datetime
import random
import gradio as gr
from glob import glob
import argparse
import itertools

# Command line argument setup
parser = argparse.ArgumentParser(description="TTS Evaluation Interface")
parser.add_argument(
    "--out_folder",
    default="./evaluation_results",
    type=str,
    help="Folder to save evaluation result per user",
)
parser.add_argument(
    "--out_file",
    default="./annotation.csv",
    type=str,
    help="File to save evaluation result",
)
parser.add_argument(
    "--sample_dirs",
    nargs="+",
    default=["model_a", 
             "model_b", 
             ],
    help="Paths to samples from different experiments/models to compare"
)
parser.add_argument(
    "--completion_code",
    default="C103IYNJ",
    type=str,
    help="Completion code for participants"
)
parser.add_argument(
    "--comparison_mode",
    default="sequential",
    choices=["pairwise", "sequential"],
    help="How to compare samples: pairwise (AB) or sequential rating"
)
parser.add_argument(
    "--shuffle",
    default=True,
    type=bool,
    help="Whether to shuffle the sample comparison order"
)
args = parser.parse_args()

# Constants
COMPLETION_CODE = args.completion_code
PROLIFIC_URL = f"https://app.prolific.com/submissions/complete?cc={COMPLETION_CODE}"

class SampleIterator:
    """Iterator for TTS samples from multiple models/experiments."""
    
    def __init__(self, sample_dirs, comparison_mode="pairwise", shuffle=True):
        """
        Initialize the sample iterator.
        
        Args:
            sample_dirs: List of directories containing samples from different models
            comparison_mode: How to compare samples ("pairwise" or "sequential")
            shuffle: Whether to shuffle the sample comparison order
        """
        self.sample_dirs = sample_dirs
        self.comparison_mode = comparison_mode
        self.should_shuffle = shuffle
        self.samples = []
        self.index = 0
        self._load_samples()
        
    def _load_samples(self):
        """Load samples from all directories and prepare comparisons."""
        # Create list of samples from each directory
        model_samples = []
        
        for directory in self.sample_dirs:
            samples = sorted(glob(os.path.join(directory, "*.wav")))
            if not samples:
                raise ValueError(f"No samples found in directory: {directory}")
            model_samples.append(samples)
        
        # Validate that all directories have the same number of samples
        sample_counts = [len(samples) for samples in model_samples]
        if len(set(sample_counts)) != 1:
            raise ValueError(f"All sample directories must have the same number of samples. Found: {sample_counts}")
        
        # Create comparisons based on mode
        if self.comparison_mode == "pairwise":
            # Create pairwise comparisons for each sample
            self.samples = []
            for i in range(len(model_samples[0])):
                # Get the i-th sample from each model
                sample_set = [model[i] for model in model_samples]
                
                # Generate all pairwise combinations
                pairs = list(itertools.combinations(range(len(self.sample_dirs)), 2))
                
                for pair in pairs:
                    self.samples.append(([sample_set[pair[0]], self.sample_dirs[pair[0]]], 
                                         [sample_set[pair[1]], self.sample_dirs[pair[1]]]))
        else:  # "sequential" mode
            # Compare all models at once for each sample
            for i in range(len(model_samples[0])):
                sample_set = []
                for j, model_dir in enumerate(self.sample_dirs):
                    sample_set.append([model_samples[j][i], model_dir])
                self.samples.append(sample_set)
        
        # Shuffle comparisons if requested
        if self.should_shuffle:
            random.shuffle(self.samples)
            
    def __iter__(self):
        """Return iterator object."""
        return self
        
    def __next__(self):
        """Get next sample comparison."""
        if self.index >= len(self.samples):
            raise StopIteration
            
        sample = self.samples[self.index]
        self.index += 1
        return sample
    
    def reset(self):
        """Reset iterator to beginning."""
        self.index = 0
        
    def peek(self):
        """Preview next sample without advancing iterator."""
        if self.index >= len(self.samples):
            return None
        return self.samples[self.index]
        
    def __len__(self):
        """Get total number of comparisons."""
        return len(self.samples)
    
    def current_index(self):
        """Get current position in iterator."""
        return self.index

class EvaluationSession:
    def __init__(self, comparison_mode="pairwise"):
        self.sample_iterator = SampleIterator(
            args.sample_dirs, 
            comparison_mode=comparison_mode,
            shuffle=args.shuffle
        )
        self.current_comparison = None
        self.out_file = ""
        self.comparison_mode = comparison_mode
    
    def reset(self):
        """Reset session to initial state."""
        self.sample_iterator.reset()
        self.current_comparison = next(self.sample_iterator)
    
    def initialize_user_session(self, name, speaks_arabic, gender):
        """Initialize a session for a specific user."""
        filename = f"{name.replace(' ', '')}_{speaks_arabic}_{gender}_eval.txt"
        self.out_file = os.path.join(args.out_folder, filename)
        
        # Create output directory
        os.makedirs(args.out_folder, exist_ok=True)
        
        # Check if user has already completed the evaluation
        if os.path.exists(self.out_file):
            with open(self.out_file, "r") as f:
                completed = len(f.readlines())
                if completed >= len(self.sample_iterator):
                    return None  # User has completed all evaluations
                    
                # Skip to the correct position in the iterator
                self.sample_iterator.reset()
                for _ in range(completed):
                    next(self.sample_iterator)
        else:
            # Create new file for user
            open(self.out_file, "a").close()
            self.sample_iterator.reset()
            
        try:
            self.current_comparison = next(self.sample_iterator)
            return True
        except StopIteration:
            return None  # No samples to evaluate
    
    def record_choice(self, choice):
        """Record user's choice and timestamp."""
        with open(self.out_file, "a") as out:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Format depends on comparison mode
            if self.comparison_mode == "pairwise":
                # For pairwise, we record which model was preferred (A or B)
                sample_a, model_a = self.current_comparison[0]
                sample_b, model_b = self.current_comparison[1]
                print(f"{timestamp},{choice},{sample_a},{sample_b},{model_a},{model_b}", file=out)
            else:
                # For "sequential" mode, record which model index was preferred
                models_info = ";".join([f"{sample},{model}" for sample, model in self.current_comparison])
                print(f"{timestamp},{choice},{models_info}", file=out)
    
    def next_sample(self):
        """Move to the next sample comparison."""
        try:
            self.current_comparison = next(self.sample_iterator)
            return False  # Not complete
        except StopIteration:
            return True  # Evaluation complete
    
    def get_current_samples(self):
        """Get current sample paths for display."""
        if self.comparison_mode == "pairwise":
            return self.current_comparison[0][0], self.current_comparison[1][0]
        else:
            # For "sequential" mode, return all samples
            return [sample for sample, _ in self.current_comparison]

    def get_progress(self):
        """Get the current progress (completed vs total)."""
        completed = self.sample_iterator.current_index() - 1  # Subtract 1 because current_index points to next sample
        total = len(self.sample_iterator)
        remaining = total - completed
        return completed, total, remaining

class BaseInterface:
    """Base class for evaluation interfaces"""
    
    def __init__(self):
        self.session = EvaluationSession(self.comparison_mode)
        self.audio_components = []
    
    def build_ui(self, blocks, user_details, begin_btn):
        """Build the UI components - to be implemented by child classes"""
        raise NotImplementedError
    
    def begin_session(self, name, speaks_arabic, gender):
        """Common logic for beginning a session"""
        if not name or not speaks_arabic or not gender:
            return [gr.Info("Please fill in all fields"), gr.Column(visible=False)] + [None] * len(self.audio_components) + [None, None]
            
        result = self.session.initialize_user_session(name, speaks_arabic, gender)
        if result is None:
            return [gr.Info("You've already completed this evaluation. If you haven't, try using a different name"), 
                    gr.Column(visible=False)] + [None] * len(self.audio_components) + [None, None]
        
        samples = self.session.get_current_samples()
        
        # Create the base return values (hiding user details, showing evaluation block)
        return_values = [gr.Column(visible=False), gr.Column(visible=True)]
        
        # Add sample paths to each audio component
        for i, component in enumerate(self.audio_components):
            if i < len(samples):
                return_values.append(samples[i])
            else:
                return_values.append(None)

        completed, total, remaining = self.session.get_progress()
        
        # Add None for preference and hide next button
        return_values.extend([None, 
                              gr.Button(visible=False),
                            #   gr.Markdown(f"**Progress:** {completed}/{total} completed ({remaining} remaining)")
                              ])
        
        return return_values
    
    def next_sample(self, *args):
        """Common logic for proceeding to the next sample"""
        self.session.record_choice(args[-3])
        is_complete = self.session.next_sample()
        
        # The last two args are preference and next_btn
        preference_component = args[-3]
        next_btn_component = args[-2]
        progress_display = args[-1]
        
        completed, total, remaining = self.session.get_progress()

        if is_complete:
            return_values =  [gr.Audio(visible=False)]*len(args)-3 + [gr.Radio(visible=False), gr.Button("Complete", visible=True, link=PROLIFIC_URL), gr.Markdown(f"**Progress:** {total}/{total} completed 0 remaining)")]
            return return_values
        
        samples = self.session.get_current_samples()
        
        # Create return values for all audio components
        return_values = []
        for i in range(len(args) - 3):  
            if i < len(samples):
                return_values.append(samples[i])
            else:
                return_values.append(None)
        # Add preference reset and hide next button
        return_values.extend([gr.Radio(value=None), 
                              gr.Button(visible=False),
                              gr.Markdown(f"**Progress:** {completed}/{total} completed ({remaining} remaining)")])
        
        return return_values

class PairwiseInterface(BaseInterface):
    """Interface for pairwise comparison of audio samples"""
    
    def __init__(self):
        self.comparison_mode = "pairwise"
        super().__init__()
    
    def build_ui(self, blocks, user_details, begin_btn):
        user_details, name, speaks_arabic, gender = user_details
        with gr.Column(visible=False) as evaluation_block:

            progress_display = gr.Markdown("**Progress:** 0/0 completed (0 remaining)")

            with gr.Row():
                audio_a = gr.Audio(type="filepath", label="Sample A", autoplay=False)
                audio_b = gr.Audio(type="filepath", label="Sample B", autoplay=False)
                self.audio_components = [audio_a, audio_b]
            
            preference = gr.Radio(
                ["A is better", "B is better", "Both are the same"],
                label="Preference",
                info="Which audio sample do you prefer?"
            )
            
            next_btn = gr.Button("Next", visible=False)
            
            # Show the Next button only after making a selection
            preference.input(lambda x: {next_btn:gr.Button(visible=True)}, outputs=[next_btn])
            
            # Process selection and move to next sample
            next_btn.click(
                self.next_sample, 
                [audio_a, audio_b, preference, next_btn, progress_display], 
                [audio_a, audio_b, preference, next_btn, progress_display]
            )
            
            # Start the evaluation process
            begin_outputs = [user_details, evaluation_block, audio_a, audio_b, preference, next_btn]
            begin_btn.click(
                self.begin_session,
                [name, speaks_arabic, gender],
                begin_outputs
            )
                
        return evaluation_block

class SequentialInterface(BaseInterface):
    """Interface for rating each audio sample one at a time"""
    
    def __init__(self):
        self.comparison_mode = "sequential"  # Use sequential mode for data loading
        super().__init__()
        self.current_index = 0
        self.total_models = len(args.sample_dirs)
    
    def build_ui(self, blocks, user_details_components, begin_btn):
        user_details, name, speaks_arabic, gender = user_details_components
        with gr.Column(visible=False) as evaluation_block:
            # Add progress display
            progress_display = gr.Markdown("**Progress:** 0/0 completed (0 remaining)")
            
            counter = gr.Markdown("<h3>Model 1 of {}</h3>".format(self.total_models))
            
            audio = gr.Audio(type="filepath", label="Current Sample", autoplay=False)
            self.audio_components = [audio]
            
            rating = gr.Radio(
                ["1 - Very Poor", "2 - Poor", "3 - Fair", "4 - Good", "5 - Excellent"],
                label="Quality Rating",
                info="Please rate the quality of this audio sample from 1-5"
            )
            
            next_btn = gr.Button("Next", visible=False)
            
            # Show the Next button only after making a selection
            rating.input(lambda x: {next_btn:gr.Button("Next",visible=True)}, outputs=[next_btn])
            
            # Process selection and move to next sample
            next_btn.click(
                self.next_sample,
                [rating, counter, progress_display],
                [audio, counter, rating, next_btn, progress_display]
            )
            
            # Start the evaluation process
            begin_outputs = [user_details, evaluation_block, audio, counter, rating, next_btn, progress_display]
            begin_btn.click(
                self.begin_session,
                [name, speaks_arabic, gender],
                begin_outputs
            )
            
        return evaluation_block
    
    def begin_session(self, name, speaks_arabic, gender):
        """Initialize session and show first sample"""
        if not name or not speaks_arabic or not gender:
            return [gr.Info("Please fill in all fields"), gr.Column(visible=False), None, None, None, None, None]
            
        result = self.session.initialize_user_session(name, speaks_arabic, gender)
        if result is None:
            return [gr.Info("You've already completed this evaluation. If you haven't, try using a different name"), 
                    gr.Column(visible=False), None, None, None, None, None]
        
        self.current_index = 0
        samples = self.session.get_current_samples()
        
        # Calculate total samples and progress
        total_utterances = len(self.session.sample_iterator)
        total_ratings = total_utterances * self.total_models
        current_utterance = self.session.sample_iterator.current_index() - 1
        completed_ratings = current_utterance * self.total_models + self.current_index
        remaining = total_ratings - completed_ratings
        
        # Return first sample only
        return [gr.Column(visible=False), gr.Column(visible=True), samples[0], 
                gr.Markdown(f"<h3>Model 1 of {self.total_models}</h3>"), None, 
                gr.Button(visible=False),
                gr.Markdown(f"**Progress:** {completed_ratings}/{total_ratings} completed ({remaining} remaining)")]
    
    def next_sample(self, rating, counter, progress_display):
        """Move to the next model or next sample set"""
        # Record the choice for the current model
        choice = f"Model {self.current_index + 1}: {rating}"
        self.session.record_choice(choice)
        
        # Move to the next model
        self.current_index += 1
        
        # Calculate total samples and progress
        total_utterances = len(self.session.sample_iterator)
        total_ratings = total_utterances * self.total_models
        current_utterance = self.session.sample_iterator.current_index() - 1
        completed_ratings = current_utterance * self.total_models + self.current_index
        remaining = total_ratings - completed_ratings
        
        # If we've seen all models in this set, move to the next sample set
        if self.current_index >= self.total_models:
            self.current_index = 0
            is_complete = self.session.next_sample()
            
            if is_complete:
                return [None, 
                        gr.Markdown("<h3>Evaluation Complete!</h3>"), 
                        gr.Radio(value=None, disabled=True), 
                        gr.Button(visible=True, link=PROLIFIC_URL),
                        gr.Markdown(f"**Evaluation Complete!** Thank you for your participation.")]
        
        # Get the current sample set
        samples = self.session.get_current_samples()
        
        # Return the appropriate model from the current sample set
        return [samples[self.current_index], 
                gr.Markdown(f"<h3>Model {self.current_index + 1} of {self.total_models}</h3>"), 
                gr.Radio(value=None), 
                gr.Button(visible=False),
                gr.Markdown(f"**Progress:** {completed_ratings}/{total_ratings} completed ({remaining} remaining)")]

def create_interface():
    """Create and configure the Gradio interface based on the comparison mode."""
    
    with gr.Blocks(theme=gr.themes.Glass()) as demo:
        gr.Markdown("## Audio Quality Evaluation")
        
        with gr.Column() as user_details:
            gr.Markdown("<h3>Please provide your details to begin</h3>")
            with gr.Row():
                name = gr.Textbox(label="Name / Prolific ID", placeholder="Enter your name")
                speaks_arabic = gr.Radio(["yes", "no"], label="Native Arabic Speaker")
                gender = gr.Radio(["Female", "Male"], label="Gender")
            begin_btn = gr.Button("Begin Evaluation", variant="primary")
        
        # Create the appropriate interface based on the comparison mode
        if args.comparison_mode == "pairwise":
            interface = PairwiseInterface()
            _ = interface.build_ui(demo, [user_details, name, speaks_arabic, gender], begin_btn)
        elif args.comparison_mode == "sequential":
            interface = SequentialInterface()
            _ = interface.build_ui(demo, [user_details, name, speaks_arabic, gender], begin_btn)
        else:
            raise ValueError(f"Unknown comparison mode: {args.comparison_mode}")
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, allowed_paths=["/"])