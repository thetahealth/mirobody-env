"""Text-only native API usage bounded by dated, verified official tariffs.

These are conservative budget charges, not account-specific USD invoices.
The reservation uses the top price tier; settlement uses the tier the measured prompt size falls
in (`PROMPT_TIERS`), the top tier when no tier table exists. Cache-hit tokens are priced at the
cached-input rate of that same tier; the receipt states the tier and rates applied.
Thinking is included exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
import urllib.request
from urllib.parse import urlparse


def utc_today():
    return datetime.now(timezone.utc).date()


#: Published cached-input rates, kept apart from `NativePrices` so the tariff fingerprint and the
#: saved batch metadata (compared on resume) do not change. Each rate is the cached-input price of
#: the same tier as the model's input price above, per million tokens in the tariff currency.
#: Checked 2026-09-29:
#:   ai.google.dev/gemini-api/docs/pricing: gemini-3.1-pro-preview "Context caching" $0.20 (prompts
#:     <= 200k) / $0.40 (> 200k) against input $2.00 / $4.00; gemini-3.7-flash "Context caching"
#:     $0.075 through 2026-12-31 against input $0.75; gemini-3.8-flash: same page (last updated
#:     2026-09-24 UTC, read 2026-09-29), the row is shared with 3.7-flash: input $0.75, "Context
#:     caching" $0.075 through 2026-12-31 ($0.15 from 2027-01-01). `cachedContentTokenCount` is part of
#:     `promptTokenCount`.
#:   help.aliyun.com/zh/model-studio/qwen3-7-flash (Beijing): "输入（缓存命中）" 0.04 / 0.12 / 0.24 CNY
#:     for input <=32k / <=256k / <=1M against input 0.2 / 0.6 / 1.2 (implicit cache hit = 20%);
#:     "显式缓存创建" is 125% of input (never requested here; a creation count in the usage is billed
#:     at that rate).
CACHE_TARIFFS = {
    ('google', 'gemini-3.1-pro-preview'): ('0.40', 'https://ai.google.dev/gemini-api/docs/pricing'),
    ('google', 'gemini-3.7-flash'): ('0.075', 'https://ai.google.dev/gemini-api/docs/pricing'),
    ('google', 'gemini-3.8-flash'): ('0.075', 'https://ai.google.dev/gemini-api/docs/pricing'),
    ('dashscope', 'qwen3.7-flash'): ('0.24', 'https://help.aliyun.com/zh/model-studio/qwen3-7-flash'),
}
CACHE_VERIFIED_ON = '2026-09-29'
DASHSCOPE_CACHE_CREATION_FACTOR = Decimal('1.25')
#: Prompt-size price tiers used at settlement, kept apart from `NativePrices` (fingerprint and saved
#: batch metadata unchanged). Each row: (highest prompt token count of the tier, input, output,
#: cached input) per million tokens in the tariff currency; the last row equals the top tier in
#: `NativePrices.verified` / `CACHE_TARIFFS`, which the reservation still uses. Thresholds are the
#: decimal reading of "k" (200k = 200000, 32k = 32000): a prompt in the 1000-vs-1024 gap lands in the
#: higher tier, never a lower one. Models absent here have a single tier. Checked 2026-09-29:
#:   ai.google.dev/gemini-api/docs/pricing, gemini-3.1-pro-preview Standard: input "$2.00, prompts <=
#:     200k tokens / $4.00, prompts > 200k tokens"; output (including thinking tokens) "$12.00 / $18.00";
#:     context caching "$0.20 / $0.40", same split. gemini-3.7/3.8-flash: one price, no split.
#:   help.aliyun.com/zh/model-studio/qwen3-7-flash, 华北 2（北京）: "输入<=32k" 0.2 / 0.8 / 缓存命中 0.04;
#:     "32k<输入<=256k" 0.6 / 2.4 / 0.12; "256k<输入<=1m" 1.2 / 4.8 / 0.24 CNY.
PROMPT_TIERS = {
    ('google', 'gemini-3.1-pro-preview'): (
        (200000, '2', '12', '0.20'), (None, '4', '18', '0.40')),
    ('dashscope', 'qwen3.7-flash'): (
        (32000, '0.2', '0.8', '0.04'), (256000, '0.6', '2.4', '0.12'), (None, '1.2', '4.8', '0.24')),
}
TIERS_VERIFIED_ON = '2026-09-29'


def prompt_tier(backend: str, model: str, prompt_tokens) -> dict | None:
    """The published price tier for a known prompt size; None = single tier or size unknown."""
    tiers = PROMPT_TIERS.get((backend, model))
    if tiers is None or type(prompt_tokens) is not int or prompt_tokens <= 0:
        return None
    for ceiling, inp, out, cache in tiers:
        if ceiling is None or prompt_tokens <= ceiling:
            return {'prompt_tokens_at_most': ceiling, 'input_per_million': inp,
                    'output_per_million': out, 'cache_input_per_million': cache,
                    'source': 'https://ai.google.dev/gemini-api/docs/pricing' if backend == 'google'
                    else 'https://help.aliyun.com/zh/model-studio/qwen3-7-flash',
                    'verified_on': TIERS_VERIFIED_ON}
    return None
#: Day the schedule below was checked against the official pages. A request is refused before
#: that day and from `VALID_DAYS` days after it on; re-check the pages and move this date.
VERIFIED_ON = '2026-09-29'
VALID_DAYS = 7


@dataclass(frozen=True)
class NativePrices:
    model: str
    backend: str
    input_per_million: str
    output_per_million: str
    currency: str
    usd_per_currency_bound: str
    max_input_tokens: int
    max_output_tokens: int
    verified_on: str
    source: str

    @classmethod
    def verified(cls, model: str, backend: str, *, today: date | None = None,
                 require_current: bool = True):
        # Only exact reviewed models/routes. No use of a nearby model's price.
        schedules = {
            ('google','gemini-3.1-pro-preview'): ('4','18','USD','1',1048576,65536,
                'https://ai.google.dev/gemini-api/docs/pricing'),
            # $0.75 / $3.75 through 2026-12-31, $1.50 / $7.50 from 2027-01-01 (pricing page).
            ('google','gemini-3.8-flash'): ('.75','3.75','USD','1',1048576,65536,
                'https://ai.google.dev/gemini-api/docs/pricing'),
            ('google','gemini-3.7-flash'): ('.75','3.75','USD','1',1048576,65536,
                'https://ai.google.dev/gemini-api/docs/pricing'),
            ('dashscope','qwen3.7-flash'): ('1.2','4.8','CNY','.20',983616,131072,
                'https://help.aliyun.com/zh/model-studio/qwen3-7-flash'),
        }
        if (backend,model) not in schedules:
            raise ValueError('No verified native tariff for the exact model and route')
        inp,out,currency,fx,context,maximum,source=schedules[(backend,model)]
        result=cls(model,backend,inp,out,currency,fx,context,maximum,VERIFIED_ON,source)
        if require_current:
            result.validate_date(today=today)
        return result

    def validate_date(self, *, today=None):
        age = ((today or utc_today()) - date.fromisoformat(self.verified_on)).days
        if not 0 <= age < VALID_DAYS:
            raise ValueError('Native tariff/FX verification expired; verify before new paid calls')

    def validate_dispatch(self):
        self.validate_date()

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()

    def upper_bound(self, prompt: str, max_output_tokens: int):
        if type(max_output_tokens) is not int or not 0<max_output_tokens<=self.max_output_tokens:
            raise ValueError('Requested native output cap exceeds the verified provider maximum')
        count=len(prompt.encode())+2048
        if count>self.max_input_tokens:
            raise ValueError('Conservative native prompt bound exceeds supported input capacity')
        return self._native_cost(count,max_output_tokens)*Decimal(self.usd_per_currency_bound)

    def _native_cost(self,prompt,output):
        return (prompt*Decimal(self.input_per_million)+output*Decimal(self.output_per_million))/1000000

    def cost_receipt(self, response: dict):
        def count(obj,key,*,required=True):
            if not required and key not in obj:return None
            value=obj.get(key)
            if type(value) is not int or value<0:
                raise ValueError('Native usage token count missing or invalid')
            return value
        if not isinstance(response,dict):raise ValueError('Missing native response')
        if self.backend=='google':
            usage=response.get('usageMetadata')
            if not isinstance(usage,dict):raise ValueError('Missing Google usageMetadata')
            prompt=count(usage,'promptTokenCount');total=count(usage,'totalTokenCount')
            candidate=count(usage,'candidatesTokenCount')
            thoughts=count(usage,'thoughtsTokenCount',required=False)
            cached=count(usage,'cachedContentTokenCount',required=False)
            tools=count(usage,'toolUsePromptTokenCount',required=False)
            output=total-prompt
            if output<0 or candidate>output or (thoughts is not None and candidate+thoughts>output):
                raise ValueError('Inconsistent Google token components')
            if tools not in (None,0):raise ValueError('Unexpected native tool charges')
            created=None
        else:
            usage=response.get('usage')
            if not isinstance(usage,dict):raise ValueError('Missing Dashscope usage')
            prompt=count(usage,'prompt_tokens');output=count(usage,'completion_tokens')
            total=count(usage,'total_tokens')
            if total!=prompt+output:raise ValueError('Inconsistent Dashscope usage totals')
            details=usage.get('prompt_tokens_details') or {}
            cached=count(details,'cached_tokens',required=False)
            created=count(details,'cache_creation_input_tokens',required=False)
            out_details=usage.get('completion_tokens_details') or {}
            thinking=count(out_details,'reasoning_tokens',required=False)
            if thinking is not None and thinking>output:raise ValueError('Reasoning exceeds total output')
        if prompt<=0 or output>self.max_output_tokens or (cached is not None and cached>prompt):
            raise ValueError('Native counts exceed the verified request envelope')
        cache_rate,cache_source=CACHE_TARIFFS[(self.backend,self.model)]
        hit=cached or 0;created=created or 0
        if hit+created>prompt:raise ValueError('Native counts exceed the verified request envelope')
        in_rate,out_rate=Decimal(self.input_per_million),Decimal(self.output_per_million)
        # The measured prompt size selects the published tier (input, output and cache together);
        # without a tier table the top tier stays. A tier never prices above the top tier.
        tier=prompt_tier(self.backend,self.model,prompt)
        if tier is not None:
            if (Decimal(tier['input_per_million'])>in_rate or Decimal(tier['output_per_million'])>out_rate
                    or Decimal(tier['cache_input_per_million'])>Decimal(cache_rate)):
                raise ValueError('Price tier exceeds the verified top tier')
            in_rate,out_rate=Decimal(tier['input_per_million']),Decimal(tier['output_per_million'])
            cache_rate=tier['cache_input_per_million']
        native=((prompt-hit-created)*in_rate+hit*Decimal(cache_rate)
                +created*in_rate*DASHSCOPE_CACHE_CREATION_FACTOR
                +output*out_rate)/1000000
        return {'kind':'published_tariff_upper_bound','basis':self.fingerprint,
                'model':self.model,'backend':self.backend,'source':self.source,
                'verified_on':self.verified_on,'actual_usd':None,
                'upper_bound_usd':str(native*Decimal(self.usd_per_currency_bound)),
                'native_currency_upper_bound':str(native),'currency':self.currency,
                'usd_per_currency_bound':self.usd_per_currency_bound,
                'prompt_tokens':prompt,'output_tokens':output,
                'cached_tokens':cached,'cache_creation_tokens':created or None,
                'cache_discount_applied':hit>0,'cache_input_per_million':cache_rate,
                'cache_source':cache_source,'cache_verified_on':CACHE_VERIFIED_ON,
                'input_per_million_applied':str(in_rate),'output_per_million_applied':str(out_rate),
                'price_tier':tier or 'top_tier'}


def preflight_native(solver, backend: str, url: str, require_current: bool = True) -> NativePrices:
    """Verify exact direct route and capacity without a generation request.

    `require_current=False` is for resuming a batch: the saved tariff is compared
    as before, and the date is enforced when a native request is dispatched.
    """
    prices=NativePrices.verified(solver.model,backend,require_current=require_current)
    allowed={'google':'generativelanguage.googleapis.com','dashscope':'dashscope.aliyuncs.com'}
    if urlparse(url).scheme!='https' or urlparse(url).hostname!=allowed[backend]:
        raise ValueError('Native tariff does not cover this endpoint')
    if solver.max_tokens>prices.max_output_tokens:
        raise ValueError('Configured native budget exceeds the provider limit; no implicit clamp')
    if backend=='google':
        req=urllib.request.Request(url.rstrip('/')+'/'+solver.model,
                                   headers={'x-goog-api-key':solver.api_key})
        try:
            with urllib.request.urlopen(req,timeout=30) as r:metadata=json.loads(r.read())
        except Exception:
            raise ValueError('Google capacity metadata cannot be verified') from None
        if (metadata.get('name')!='models/'+solver.model
                or metadata.get('outputTokenLimit')!=prices.max_output_tokens
                or metadata.get('inputTokenLimit')!=prices.max_input_tokens):
            raise ValueError('Google live model capacity differs from the verified record')
    return prices
