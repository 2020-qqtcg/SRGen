"""
TNOT (Test-time Training) Universal Decorator

This module provides a universal decorator that can add TNOT functionality to any 
Transformers CausalLM model without requiring separate modeling files for each model type.

Usage:
    from TNOT.tnot_decorator import enable_tnot
    from transformers import AutoModelForCausalLM
    
    # Apply TNOT decorator to any model class
    TNOTModelClass = enable_tnot(AutoModelForCausalLM)
    model = TNOTModelClass.from_pretrained("model_name")
    
    # Or apply to specific model classes
    @enable_tnot
    class MyCustomModel(LlamaForCausalLM):
        pass
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Optional, Union, Tuple, List
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
import functools


def enable_tnot(model_class):
    """
    Decorator that adds TNOT (Test-time Training) functionality to any CausalLM model class.
    
    Args:
        model_class: A Transformers CausalLM model class (e.g., LlamaForCausalLM, GPT2LMHeadModel, etc.)
                    or AutoModelForCausalLM factory class
        
    Returns:
        Enhanced model class with TNOT capabilities
    """
    
    # Handle AutoModelForCausalLM factory class specially
    if not hasattr(model_class, 'forward'):
        return _create_tnot_auto_model_class(model_class)
    
    # Store original methods for regular model classes
    original_init = model_class.__init__
    original_forward = model_class.forward
    
    def enhanced_init(self, config, *args, **kwargs):
        """Enhanced __init__ that adds TNOT attributes"""
        # Call original __init__
        original_init(self, config, *args, **kwargs)
        
        # Initialize TNOT-specific attributes
        self.delta = None
        self.high_entropy_detected = False
        self.high_entropy_position = None
        self.entropy_threshold = None
        self.entropy_history = []
        self.index = None
    
    def enhanced_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        masked_token_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Enhanced forward method with TNOT functionality"""
        
        # Handle default values like in original implementation
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Call the base model (not the full forward) - this is the key difference!
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        
        # Extract hidden states - consistent with original implementation
        hidden_states = outputs[0]
        
        # Store original hidden states for entropy comparison
        original_hidden_states = hidden_states.clone()

        prompt_only = os.environ.get("prompt_only", "False") == "True" 
        stage = "prompt" if prompt_only else "generation"
        
        # Apply TNOT logic
        hidden_states = apply_tnot_logic(
            self, 
            hidden_states, 
            input_ids, 
            masked_token_ids,
            prompt_only
        )
        
        # Handle entropy recording and analysis
        handle_entropy_analysis(
            self, 
            original_hidden_states, 
            hidden_states, 
            input_ids, 
            logits_to_keep
        )
        
        # Recompute logits with modified hidden states
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        # Apply entropy-based early stopping if enabled
        logits = apply_entropy_control(
            self, 
            logits, 
            past_key_values, 
            input_ids,
            logits_to_keep,
            stage
        )
        
        # Handle loss computation - exactly like original implementation
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        # Return in the same format as original implementation
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )
    
    # Replace methods in the class using module-level functions
    model_class.__init__ = enhanced_init
    model_class.forward = enhanced_forward
    model_class.reset_entropy_detection = _reset_entropy_detection_method
    model_class.reset_model_parameters = _reset_model_parameters_method
    model_class._safe_decode_token = _safe_decode_token_method
    model_class._safe_decode_sequence = _safe_decode_sequence_method
    
    return model_class


def apply_tnot_logic(model, hidden_states, input_ids, masked_token_ids, prompt_only):
    """
    Apply TNOT (Test-time Training) logic to hidden states
    
    Args:
        model: The model instance
        hidden_states: Current hidden states
        input_ids: Input token IDs
        masked_token_ids: Token IDs to mask during training
        
    Returns:
        Modified hidden states after TNOT processing
    """
    prompt_only = os.environ.get("prompt_only", "False") == "True" 
    stage = "prompt" if prompt_only else "generation"
    
    if prompt_only:
        if model.delta is not None:
            # Apply existing delta but don't modify hidden_states yet
            pass
            
        times = int(os.environ.get("times", 1))
        lr = float(os.environ.get("lr", 0.1))
        
        with torch.enable_grad():
            
            if model.delta is not None:
                delta_high = nn.Parameter(0.0 * torch.randn([1, 1, hidden_states.shape[-1]]).to(hidden_states))
                # Optimize delta_high with joint loss (CE + entropy)
                optimizer_high = torch.optim.AdamW([delta_high], lr=lr, weight_decay=1e-8, eps=1e-5)
                for _ in range(times):
                    optimizer_high.zero_grad()
                    transformed_hidden = hidden_states + delta_high

                    logits = model.lm_head(transformed_hidden)
                    loss_fct = nn.CrossEntropyLoss()
                    shift_logits = logits[..., :-1, :].contiguous()
                    
                    # Use prompt as labels
                    shift_labels = input_ids[:, 1:].contiguous()
                    shift_labels = shift_labels.to(shift_logits.device)

                    # Apply the mask to the labels
                    masked_labels = shift_labels.clone()
                    if masked_token_ids is not None:
                        for token_id in masked_token_ids:
                            masked_labels[masked_labels == token_id] = -100
                    
                    ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), masked_labels.view(-1))
                    
                    # Add entropy loss for the last position
                    last_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
                    last_probs = F.softmax(last_logits, dim=-1)
                    entropy = -torch.sum(last_probs * torch.log(last_probs + 1e-10), dim=-1)  # Shape: [batch_size]
                    entropy_loss = torch.mean(entropy)  # Average over batch
                    
                    # Combine losses using weighted average
                    entropy_weight = float(os.environ.get("entropy_weight", "0.1"))
                    loss = (1 - entropy_weight) * ce_loss + entropy_weight * entropy_loss
                    
                    loss.backward()
                    optimizer_high.step()

                # Apply delta_high for current prompt processing
                # Note: This modifies hidden_states only during prompt stage when delta already exists
                hidden_states = hidden_states + delta_high
            
            # Optimize delta_normal with only cross-entropy loss
            else:
                delta_normal = nn.Parameter(0.0 * torch.randn([1, 1, hidden_states.shape[-1]]).to(hidden_states))
                optimizer_normal = torch.optim.AdamW([delta_normal], lr=lr, weight_decay=1e-8, eps=1e-5)
                for _ in range(times):
                    optimizer_normal.zero_grad()
                    transformed_hidden = hidden_states + delta_normal
                    logits = model.lm_head(transformed_hidden)
                    loss_fct = nn.CrossEntropyLoss()
                    shift_logits = logits[..., :-1, :].contiguous()
                    
                    # Use prompt as labels
                    shift_labels = input_ids[:, 1:].contiguous()
                    shift_labels = shift_labels.to(shift_logits.device)

                    # Apply the mask to the labels
                    masked_labels = shift_labels.clone()
                    if masked_token_ids is not None:
                        for token_id in masked_token_ids:
                            masked_labels[masked_labels == token_id] = -100

                    ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), masked_labels.view(-1))
                    
                    # Only use cross-entropy loss for delta_normal
                    loss = ce_loss
                    
                    loss.backward()
                    optimizer_normal.step()
                
                # Store delta_normal for subsequent generation stages
                model.delta = delta_normal.detach().clone()

                # Note: In the original implementation, delta is not applied to hidden_states
                # at the end of prompt stage. It's only used during optimization.
                # hidden_states = hidden_states + model.delta
        
        os.environ["prompt_only"] = "False"
        torch.cuda.empty_cache()
        
    else:
        if model.delta is not None:
            # Apply delta_normal (cross-entropy optimized) for generation
            # Note: In the original implementation, this was commented out
            # hidden_states = hidden_states + model.delta
            pass
    
    return hidden_states


def handle_entropy_analysis(model, original_hidden_states, modified_hidden_states, input_ids, logits_to_keep):
    """Handle entropy recording and analysis"""
    
    # Calculate entropy and record analysis if enabled
    if os.environ.get("record_entropy", "False") == "True" and model.delta is not None:
        _record_entropy_analysis(model, original_hidden_states, modified_hidden_states, input_ids, logits_to_keep)

    if response_entropy_file := os.environ.get("response_entropy_file", ""):
        _record_response_entropy(model, original_hidden_states, modified_hidden_states, input_ids, logits_to_keep, response_entropy_file)


def apply_entropy_control(model, logits, past_key_values, input_ids, logits_to_keep=0, stage="generation"):
    """Apply entropy-based early stopping logic"""
    
    # Add entropy-based early stopping logic
    entropy_control_enabled = os.environ.get("entropy_control", "False") == "True"
    if entropy_control_enabled and logits.shape[1] > 0 and stage == "generation":
        entropy_threshold = float(os.environ.get("entropy_threshold", "3.0"))
        
        # Calculate entropy for the last token position
        last_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
        last_probs = F.softmax(last_logits, dim=-1)
        entropy = -torch.sum(last_probs * torch.log(last_probs + 1e-10), dim=-1)  # Shape: [batch_size]

        # Dynamic entropy threshold
        if os.environ.get("adaptive_entropy", "False") == "True":
            adaptive_entropy_N = int(os.environ.get("adaptive_entropy_N", "20"))
            adaptive_entropy_K = float(os.environ.get("adaptive_entropy_K", "2"))
            current_len = len(model.entropy_history) + 1

            # Only calculate dynamic entropy threshold if we have enough history
            if current_len > adaptive_entropy_N:
                window = torch.tensor(model.entropy_history[-adaptive_entropy_N:], device=entropy.device)

                mean_history = torch.mean(window)
                std_history = max(torch.std(window), 0.5)

                entropy_threshold = mean_history + adaptive_entropy_K * std_history
                entropy_threshold = entropy_threshold.item()
                entropy_threshold = max(entropy_threshold, 1.8)  # Ensure non-negative threshold

            model.entropy_history.append(entropy.item())
        
        # Check if entropy exceeds threshold
        high_entropy_mask = entropy > entropy_threshold
        
        if high_entropy_mask.any():
            # Mark that high entropy was detected
            model.high_entropy_detected = True
            # Get the current sequence length for position tracking
            if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
                current_length = past_key_values.get_seq_length() + logits.shape[1]
            elif input_ids is not None:
                current_length = input_ids.shape[1]
            else:
                current_length = 0
            model.high_entropy_position = current_length
            
            # Force EOS token for high entropy samples
            eos_token_id = getattr(model.config, 'eos_token_id', 2)  # Default to 2 if not specified
            
            # Create new logits with EOS token as the highest probability
            modified_logits = logits.clone()

            dtype = modified_logits.dtype
            large_value = 1e4
            if os.environ.get("log_entropy_control", "False") == "True":
                print(f"large_value: {large_value:.4f} for dtype {dtype}")

            for batch_idx in range(logits.shape[0]):
                if high_entropy_mask[batch_idx]:
                    # Set EOS token to very high logit value
                    modified_logits[batch_idx, -1, :] = -large_value
                    modified_logits[batch_idx, -1, eos_token_id] = large_value
            
            logits = modified_logits
            
            # Log entropy detection
            if os.environ.get("log_entropy_control", "False") == "True":
                for batch_idx in range(entropy.shape[0]):
                    if high_entropy_mask[batch_idx]:
                        print(f"High entropy detected: {entropy[batch_idx].item():.4f} > {entropy_threshold} at position {current_length}")
    
    return logits


def _record_entropy_analysis(model, original_hidden_states, modified_hidden_states, input_ids, logits_to_keep):
    """Record entropy analysis for tokens before and after applying delta"""
    try:
        # Calculate logits for both original and modified hidden states
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        
        with torch.no_grad():
            original_logits = model.lm_head(original_hidden_states[:, slice_indices, :])
            modified_logits = model.lm_head(modified_hidden_states[:, slice_indices, :])
            
            # Calculate probabilities and entropy
            original_probs = F.softmax(original_logits, dim=-1)
            modified_probs = F.softmax(modified_logits, dim=-1)
            
            # Calculate entropy: -sum(p * log(p))
            original_entropy = -torch.sum(original_probs * torch.log(original_probs + 1e-10), dim=-1)
            modified_entropy = -torch.sum(modified_probs * torch.log(modified_probs + 1e-10), dim=-1)
            
            # Get predicted tokens
            original_tokens = torch.argmax(original_logits, dim=-1)
            modified_tokens = torch.argmax(modified_logits, dim=-1)
            
            # Process each batch and sequence position
            batch_size, seq_len = original_tokens.shape
            
            # Get output file path
            output_file = os.environ.get("entropy_output_file", "entropy_analysis.jsonl")
            
            # Prepare data for logging
            entropy_data = []
            
            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    # Get the actual position in the full sequence
                    if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                        # Only looking at last logits_to_keep tokens
                        actual_seq_idx = input_ids.shape[1] - logits_to_keep + seq_idx
                    else:
                        actual_seq_idx = seq_idx
                        
                    # Skip if out of bounds
                    if actual_seq_idx >= input_ids.shape[1]:
                        continue
                        
                    # Get input token (the token that produced this prediction)
                    input_token = input_ids[batch_idx, actual_seq_idx].item()
                    
                    record = {
                        "batch_idx": batch_idx,
                        "seq_idx": actual_seq_idx,
                        "input_token": input_token,
                        "input_token_decoded": model._safe_decode_token(input_token),
                        "original_predicted_token": original_tokens[batch_idx, seq_idx].item(),
                        "original_predicted_decoded": model._safe_decode_token(original_tokens[batch_idx, seq_idx].item()),
                        "original_entropy": original_entropy[batch_idx, seq_idx].item(),
                        "modified_predicted_token": modified_tokens[batch_idx, seq_idx].item(),
                        "modified_predicted_decoded": model._safe_decode_token(modified_tokens[batch_idx, seq_idx].item()),
                        "modified_entropy": modified_entropy[batch_idx, seq_idx].item(),
                        "entropy_diff": (modified_entropy[batch_idx, seq_idx] - original_entropy[batch_idx, seq_idx]).item(),
                    }
                    entropy_data.append(record)
            
            # Write to file
            with open(output_file, "a", encoding="utf-8") as f:
                for record in entropy_data:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
    except Exception as e:
        # Log error but don't interrupt the forward pass
        print(f"Error in entropy analysis: {e}")


def _record_response_entropy(model, original_hidden_states, modified_hidden_states, input_ids, logits_to_keep, response_entropy_file):
    """Record entropy analysis for response tokens only"""
    try:
        # Calculate logits for both original and modified hidden states
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        
        with torch.no_grad():
            original_logits = model.lm_head(original_hidden_states[:, slice_indices, :])
            modified_logits = model.lm_head(modified_hidden_states[:, slice_indices, :])
            
            # Calculate probabilities and entropy
            original_probs = F.softmax(original_logits, dim=-1)
            modified_probs = F.softmax(modified_logits, dim=-1)
            
            # Calculate entropy: -sum(p * log(p))
            original_entropy = -torch.sum(original_probs * torch.log(original_probs + 1e-10), dim=-1)
            modified_entropy = -torch.sum(modified_probs * torch.log(modified_probs + 1e-10), dim=-1)
            
            # Get predicted tokens
            original_tokens = torch.argmax(original_logits, dim=-1)
            modified_tokens = torch.argmax(modified_logits, dim=-1)
            
            # Process each batch and sequence position
            batch_size, seq_len = original_tokens.shape
            
            # Prepare response data list
            response_data = []
            
            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    # Create record for each response token (index starts from 0)
                    model.index = 0 if model.index is None else model.index + 1
                    record = {
                        "token_index": model.index,  # Response token index starting from 0
                        "original_entropy": original_entropy[batch_idx, seq_idx].item(),
                        "modified_entropy": modified_entropy[batch_idx, seq_idx].item(),
                        "original_token_decoded": model._safe_decode_token(original_tokens[batch_idx, seq_idx].item()),
                        "modified_token_decoded": model._safe_decode_token(modified_tokens[batch_idx, seq_idx].item()),
                    }
                    response_data.append(record)
            
            # Read existing data if file exists
            existing_data = []
            if os.path.exists(response_entropy_file):
                try:
                    with open(response_entropy_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = []
            
            # Extend existing data with new response data
            existing_data.extend(response_data)
            
            # Write updated data back to file
            with open(response_entropy_file, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
                    
    except Exception as e:
        # Log error but don't interrupt the forward pass
        print(f"Error in response entropy recording: {e}")


def _create_tnot_auto_model_class(auto_model_class):
    """
    Create a TNOT-enabled wrapper for AutoModelForCausalLM factory class.
    
    Args:
        auto_model_class: AutoModelForCausalLM class
        
    Returns:
        TNOT-enabled wrapper class
    """
    
    class TNOTAutoModelForCausalLM:
        """TNOT-enabled wrapper for AutoModelForCausalLM"""
        
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Load model and apply TNOT functionality"""
            # Load the actual model using the original AutoModelForCausalLM
            model = auto_model_class.from_pretrained(*args, **kwargs)
            
            # Apply TNOT functionality to the loaded model instance
            model = _apply_tnot_to_instance(model)
            
            return model
        
        # Forward other class methods to the original class
        def __getattr__(self, name):
            return getattr(auto_model_class, name)
    
    return TNOTAutoModelForCausalLM


def _enhanced_forward_for_instance(
    model,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    masked_token_ids: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    """Enhanced forward method with TNOT functionality for model instances"""
    
    # Handle default values like in original implementation
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else model.config.use_cache
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict
    
    # Prepare arguments for original forward method
    forward_kwargs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'past_key_values': past_key_values,
        'inputs_embeds': inputs_embeds,
        'use_cache': use_cache,
        'output_attentions': output_attentions,
        'output_hidden_states': output_hidden_states,
        'return_dict': return_dict,
        **kwargs,
    }
    
    # Add cache_position if supported by the model
    import inspect
    original_forward = getattr(model.__class__, 'forward', None)
    if original_forward and 'cache_position' in inspect.signature(original_forward).parameters:
        forward_kwargs['cache_position'] = cache_position
    
    # Call the underlying model's forward method (self.model for CausalLM models)
    outputs = model.model(**forward_kwargs)
    
    # Extract hidden states - consistent with original implementation
    hidden_states = outputs[0]
    original_hidden_states = hidden_states.clone()

    prompt_only = os.environ.get("prompt_only", "False") == "True" 
    stage = "prompt" if prompt_only else "generation"
    
    # Apply TNOT logic
    hidden_states = apply_tnot_logic(model, hidden_states, input_ids, masked_token_ids, prompt_only)
    
    # Handle entropy analysis and recording
    handle_entropy_analysis(model, original_hidden_states, hidden_states, input_ids, logits_to_keep)
    
    # Recompute logits with modified hidden states
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = model.lm_head(hidden_states[:, slice_indices, :])
    
    # Apply entropy-based early stopping if enabled
    logits = apply_entropy_control(
        model, 
        logits, 
        past_key_values, 
        input_ids,
        logits_to_keep,
        stage
    )
    
    # Handle loss computation - exactly like original implementation
    loss = None
    if labels is not None:
        loss = model.loss_function(logits=logits, labels=labels, vocab_size=model.config.vocab_size, **kwargs)
    
    # Return in the same format as original implementation
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output
    
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
        hidden_states=outputs.hidden_states if output_hidden_states else None,
        attentions=outputs.attentions if output_attentions else None,
    )


def _apply_tnot_to_instance(model):
    """
    Apply TNOT functionality to an already instantiated model.
    
    Args:
        model: An instantiated CausalLM model
        
    Returns:
        The same model with TNOT functionality added
    """
    
    # Initialize TNOT-specific attributes
    model.delta = None
    model.high_entropy_detected = False
    model.high_entropy_position = None
    model.entropy_threshold = None
    model.entropy_history = []
    model.index = None
    
    # Replace the model's forward method with our enhanced version
    # Use functools.partial to create a pickleable bound method
    model.forward = functools.partial(_enhanced_forward_for_instance, model)
    
    # Add the missing TNOT methods
    _add_tnot_methods(model)
    
    return model


def _reset_entropy_detection(model):
    """Reset entropy detection state for new generation"""
    model.high_entropy_detected = False
    model.high_entropy_position = None


def _reset_model_parameters(model):
    """Reset model parameters"""
    model.delta = None
    model.entropy_threshold = None
    model.entropy_history = []


def _safe_decode_token(token_id):
    """Safely decode a token ID to text, handling potential errors"""
    try:
        # Try to get tokenizer from the model
        tokenizer = AutoTokenizer.from_pretrained(os.environ.get("tokenizer_path"))
        # Decode the token
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        # Clean up the decoded text (remove extra spaces, special formatting)
        decoded = decoded.strip()
        if not decoded:  # If empty after stripping
            decoded = tokenizer.convert_ids_to_tokens([token_id])[0]
        return decoded
    except Exception as e:
        return f"<decode_error_{token_id}>"


def _safe_decode_sequence(token_ids):
    """Safely decode a sequence of token IDs to text"""
    try:
        # Try to get tokenizer from the model
        tokenizer = AutoTokenizer.from_pretrained(os.environ.get("tokenizer_path"))
        # Decode the sequence
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        return decoded
    except Exception as e:
        return f"<decode_error: {e}>"


def _record_response_entropy(model, original_hidden_states, modified_hidden_states, input_ids, logits_to_keep, response_entropy_file):
    """Record response entropy data - implementation matches original"""
    try:
        # Get tokenizer path
        tokenizer_path = os.environ.get("tokenizer_path")
        if not tokenizer_path:
            return
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Calculate entropy for both original and modified states
        original_logits = model.lm_head(original_hidden_states)
        modified_logits = model.lm_head(modified_hidden_states)
        
        # Calculate entropies
        original_probs = F.softmax(original_logits, dim=-1)
        modified_probs = F.softmax(modified_logits, dim=-1)
        
        original_entropy = -torch.sum(original_probs * torch.log(original_probs + 1e-8), dim=-1)
        modified_entropy = -torch.sum(modified_probs * torch.log(modified_probs + 1e-8), dim=-1)
        
        # Prepare response data
        response_data = []
        
        batch_size, seq_len = input_ids.shape
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        
        for batch_idx in range(batch_size):
            for pos in range(seq_len):
                if slice_indices == slice(None) or pos >= seq_len + slice_indices.start:
                    token_id = input_ids[batch_idx, pos].item()
                    
                    response_entry = {
                        "index": getattr(model, 'index', 0),
                        "batch_idx": batch_idx,
                        "position": pos,
                        "token_id": token_id,
                        "token_text": _safe_decode_token(token_id),
                        "original_entropy": original_entropy[batch_idx, pos].item(),
                        "modified_entropy": modified_entropy[batch_idx, pos].item(),
                        "entropy_diff": (modified_entropy[batch_idx, pos] - original_entropy[batch_idx, pos]).item(),
                        "timestamp": datetime.now().isoformat()
                    }
                    response_data.append(response_entry)
        
        # Load existing data and append new data
        existing_data = []
        if os.path.exists(response_entropy_file):
            try:
                with open(response_entropy_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []
        
        # Extend existing data with new response data
        existing_data.extend(response_data)
        
        # Write updated data back to file
        with open(response_entropy_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
                
    except Exception as e:
        # Log error but don't interrupt the forward pass
        print(f"Error in response entropy recording: {e}")


def _reset_entropy_detection_method(self):
    """Method wrapper for reset_entropy_detection"""
    return _reset_entropy_detection(self)


def _reset_model_parameters_method(self):
    """Method wrapper for reset_model_parameters"""
    return _reset_model_parameters(self)


def _safe_decode_token_method(self, token_id):
    """Method wrapper for _safe_decode_token"""
    return _safe_decode_token(token_id)


def _safe_decode_sequence_method(self, token_ids):
    """Method wrapper for _safe_decode_sequence"""
    return _safe_decode_sequence(token_ids)


def _add_tnot_methods(model):
    """Add all the TNOT-specific methods to the model instance"""
    
    # Use functools.partial to create pickleable bound methods
    model.reset_entropy_detection = functools.partial(_reset_entropy_detection, model)
    model.reset_model_parameters = functools.partial(_reset_model_parameters, model)
    model._safe_decode_token = _safe_decode_token  # This doesn't need the model
    model._safe_decode_sequence = _safe_decode_sequence  # This doesn't need the model
    model._record_response_entropy = functools.partial(_record_response_entropy, model)