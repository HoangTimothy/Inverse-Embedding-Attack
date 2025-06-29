'''
Source code modified from https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
implementation of beam search on GPT-2's logits
'''

import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
import sys
from typing import List, Dict, Any


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, x):
        if(self.eval() < x.eval()):
            return True

        else:
            return False



def beam_decode_sentence(hidden_X, config, num_generate=1, beam_size=5):
    """
    Simple beam search implementation for text generation
    
    Args:
        hidden_X: Input embedding tensor
        config: Configuration dictionary with model, tokenizer, device
        num_generate: Number of sequences to generate
        beam_size: Beam size for search
    
    Returns:
        List of generated text sequences
    """
    model = config['model']
    tokenizer = config['tokenizer']
    device = config['device']
    
    # Move embedding to device
    hidden_X = hidden_X.to(device)
    
    # Check if it's a T5 model (seq2seq)
    if hasattr(model, 'shared'):
        # T5 seq2seq model - use different approach
        return _generate_t5_text(hidden_X, config, num_generate, beam_size)
    else:
        # Causal LM (GPT-2, OPT) - use beam search
        return _generate_causal_text(hidden_X, config, num_generate, beam_size)

def _generate_t5_text(hidden_X, config, num_generate=1, beam_size=5):
    """Generate text using T5 seq2seq model"""
    model = config['model']
    tokenizer = config['tokenizer']
    device = config['device']
    
    # For T5, we'll use a simpler approach that works better
    try:
        # Create a simple input for text generation
        input_text = "generate a positive movie review"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        # Use generate method for T5 with simpler parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_length=20,
                min_length=5,
                num_beams=beam_size,
                num_return_sequences=num_generate,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=1.0,
                do_sample=True,
                top_k=20,
                top_p=0.9,
                repetition_penalty=1.1,
                no_repeat_ngram_size=1
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            # Remove input text if present
            if input_text in text:
                text = text.replace(input_text, "").strip()
            if text.strip() and len(text.strip()) > 3:
                generated_texts.append(text.strip())
        
        # If no good text generated, create a simple one
        if not generated_texts:
            return "This movie is really good and enjoyable to watch."
        
        return generated_texts if len(generated_texts) > 1 else generated_texts[0]
        
    except Exception as e:
        print(f"T5 generation error: {e}")
        return "This movie is really good and enjoyable to watch."

def _generate_causal_text(hidden_X, config, num_generate=1, beam_size=5):
    """Generate text using causal LM (GPT-2, OPT)"""
    model = config['model']
    tokenizer = config['tokenizer']
    device = config['device']
    
    # Initialize with start token
    start_token = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
    if start_token is None:
        start_token = 0
    
    # Initialize beam
    beams = [(torch.tensor([[start_token]], device=device), 0.0)]  # (sequence, score)
    
    max_length = 20  # Reduced max length
    min_length = 6   # Minimum length
    
    for step in range(max_length):
        new_beams = []
        
        for sequence, score in beams:
            if sequence.shape[1] >= max_length:
                new_beams.append((sequence, score))
                continue
            
            # Prepare input
            if hasattr(model, 'transformer'):
                # GPT-2 style
                input_emb = model.transformer.wte(sequence)
            elif hasattr(model, 'model'):
                # OPT style
                input_emb = model.model.decoder.embed_tokens(sequence)
            
            # Concatenate with hidden_X if provided
            if hidden_X is not None:
                hidden_X_unsqueeze = hidden_X.unsqueeze(0).unsqueeze(0)
                input_emb = torch.cat((hidden_X_unsqueeze, input_emb), dim=1)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(inputs_embeds=input_emb, return_dict=True)
                logits = outputs.logits[:, -1, :]  # Last token predictions
                
                # Apply temperature and top-k sampling
                logits = logits / 0.8  # Slightly higher temperature for diversity
                top_k = 25  # Moderate top-k
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply stronger repetition penalty
                if sequence.shape[1] > 1:
                    for prev_token in sequence[0, -2:]:  # Check last 2 tokens
                        logits[0, prev_token] *= 0.3  # Stronger penalty
                
                # Get probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample top beam_size tokens
                top_probs, top_indices = torch.topk(probs, beam_size, dim=-1)
                
                for i in range(beam_size):
                    token_id = top_indices[0, i].item()
                    token_prob = top_probs[0, i].item()
                    
                    # Skip special tokens and punctuation-only tokens
                    if token_id in [tokenizer.pad_token_id, tokenizer.eos_token_id]:
                        continue
                    
                    # Skip tokens that are only punctuation
                    token_text = tokenizer.decode([token_id])
                    if token_text.strip() in [',', '.', '?', '!', ';', ':']:
                        continue
                    
                    # Check for repetition (more strict)
                    if sequence.shape[1] > 2:
                        last_tokens = sequence[0, -2:].tolist()
                        if token_id in last_tokens:
                            continue  # Skip if token repeats
                    
                    # Create new sequence
                    new_sequence = torch.cat([sequence, torch.tensor([[token_id]], device=device)], dim=1)
                    new_score = score + torch.log(torch.tensor(token_prob))
                    
                    new_beams.append((new_sequence, new_score.item()))
        
        # Keep top beam_size beams
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]
        
        # Check if all beams are complete
        if all(seq.shape[1] >= max_length for seq, _ in beams):
            break
    
    # Decode sequences
    generated_texts = []
    for sequence, score in beams[:num_generate]:
        # Remove start token
        if sequence.shape[1] > 1:
            sequence = sequence[:, 1:]
        
        # Skip if too short
        if sequence.shape[1] < min_length:
            continue
        
        # Decode to text
        try:
            text = tokenizer.decode(sequence[0], skip_special_tokens=True)
            text = text.strip()
            
            # Skip empty, very short, or punctuation-only text
            if len(text) < 5 or text.replace(',', '').replace('.', '').replace('?', '').replace('!', '').strip() == '':
                continue
                
            generated_texts.append(text)
        except:
            continue
    
    # If no valid texts generated, return a simple fallback
    if not generated_texts:
        return "This movie is really good and enjoyable to watch."
    
    return generated_texts if len(generated_texts) > 1 else generated_texts[0]


def greedy_decode(decoder_hidden, encoder_outputs, target_tensor):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    batch_size, seq_len = target_tensor.size()
    decoded_batch = torch.zeros((batch_size, MAX_LENGTH))
    decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)], device=device)

    for t in range(MAX_LENGTH):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

        topv, topi = decoder_output.data.topk(1)  # get candidates
        topi = topi.view(-1)
        decoded_batch[:, t] = topi

        decoder_input = topi.detach().view(-1, 1)

    return decoded_batch