"""Backend-only Responses API integration. Never logs media or provider bodies."""
import json
import os
import urllib.request
import urllib.error
from neck_v1_scoring import ModelOutput, RUBRIC

PROMPT_VERSION = 'neck-rubric-v1.5'
DEFAULT_MODEL = 'gpt-4.1-2025-04-14'
# A real ten-image response took 22s; the previous 15s socket timeout
# discarded valid responses. Keep one attempt, with a matching larger UI bound.
PROVIDER_TIMEOUT_SECONDS = 45

class NeckProviderError(RuntimeError):
    pass


def interpret(context, images, transport=None):
    key = os.environ.get('OPENAI_API_KEY')
    if not key and transport is None:
        raise NeckProviderError('Neck analysis requires server OPENAI_API_KEY')
    model = os.environ.get('NECK_OPENAI_MODEL', DEFAULT_MODEL)
    content = [{'type': 'input_text', 'text': json.dumps(context, allow_nan=False)}]
    for item in images:
        content.extend([{'type':'input_text','text':f"Frame {item['index']} at {item['t']:.3f}s, {item['kind']}"},
                        {'type':'input_image','image_url':item['url'],'detail':'high'}])
    payload = dict(model=model, store=False, temperature=0,
                   input=[{'role':'system','content':RUBRIC}, {'role':'user','content':content}],
                   text={'format':{'type':'json_schema','name':'neck_dimensions_v1','strict':True,
                                   'schema':ModelOutput.model_json_schema()}}, max_output_tokens=1600)
    try:
        if transport is not None:
            response = transport(payload)
        else:
            request=urllib.request.Request('https://api.openai.com/v1/responses',
                    data=json.dumps(payload).encode(),method='POST',
                    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=PROVIDER_TIMEOUT_SECONDS) as raw:
                response=json.load(raw)
        if response.get('status') != 'completed':
            raise NeckProviderError('Neck analysis provider did not complete')
        texts=[]
        for item in response.get('output',[]):
            for block in item.get('content',[]):
                if block.get('type')=='refusal':
                    raise NeckProviderError('Neck analysis provider declined the request')
                if block.get('type')=='output_text':
                    texts.append(block['text'])
        if len(texts)!=1:
            raise NeckProviderError('Neck analysis provider returned invalid structured output')
        return ModelOutput.model_validate_json(texts[0])
    except urllib.error.HTTPError as exc:
        messages = {401: 'OpenAI authentication failed (HTTP 401)',
                    403: 'OpenAI access denied (HTTP 403)',
                    429: 'OpenAI quota or rate limit reached (HTTP 429)'}
        # Preserve only documented error identifiers; never expose provider messages/bodies.
        allowed = {'credit_balance_exhausted', 'insufficient_quota',
                   'organization_spend_limit_exceeded', 'project_spend_limit_exceeded',
                   'organization_usage_limit_exceeded', 'rate_limit_exceeded', 'slow_down'}
        identifier = None
        if exc.code == 429:
            try:
                body = json.loads(exc.read(16384))
                error = body.get('error', {}) if isinstance(body, dict) else {}
                if isinstance(error, dict):
                    code, kind = error.get('code'), error.get('type')
                    if isinstance(code, str) and code in allowed:
                        identifier = code
                    elif kind == 'insufficient_quota':
                        identifier = 'insufficient_quota'
            except (ValueError, OSError):
                pass
        message = messages.get(exc.code, f'OpenAI service error (HTTP {exc.code})')
        if identifier:
            message += f' [{identifier}]'
        raise NeckProviderError(message) from exc
    except TimeoutError:
        raise NeckProviderError('OpenAI analysis timed out') from None
    except NeckProviderError:
        raise
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        raise NeckProviderError('Neck analysis provider failed or returned invalid structured output') from exc
