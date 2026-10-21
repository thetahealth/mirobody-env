"""FHIR R4 REST projection of a single haenv tool session (stdlib only).

Day zero is 2000-01-01 unless an anchor is supplied. The public extension namespace
is http://haenv.example/fhir/StructureDefinition/. `case-day` records the source
field and integer day; `tool-fields` preserves fields with no standard FHIR slot
as JSON. `from_fhir` reconstructs the exact function-call result, including compact
stream entries. Calendar dates, rather than the preservation JSON, carry times.

Sources are a DocumentReference inventory, protocols are PlanDefinition resources,
and menu entries are ActivityDefinition resources. Legacy rest_form operations
are aliases. A FHIR batch Bundle at POST / executes purchases from one tool turn
as one purchase round, in entry order, just as the gated query loop does.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

from haenv.interaction.query import fhir
from haenv.serve.session import ToolSession

NAMESPACE = 'http://haenv.example/fhir/StructureDefinition/'
CASE_DAY = NAMESPACE + 'case-day'
FIELDS = NAMESPACE + 'tool-fields'
DEFAULT_ANCHOR = '2000-01-01'


class DayMap:
    """An exactly invertible case-day / proleptic Gregorian date mapping."""

    def __init__(self, anchor=DEFAULT_ANCHOR):
        self.anchor = date.fromisoformat(anchor)

    def to_date(self, day):
        if isinstance(day, bool) or not isinstance(day, int):
            raise ValueError('A case day must be an integer')
        return (self.anchor + timedelta(days=day)).isoformat()

    def to_day(self, value):
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            raise ValueError('Use an ISO date (YYYY-MM-DD)')
        return (date.fromisoformat(value) - self.anchor).days


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _id(value):
    value = str(value)
    return value if re.fullmatch(r'[A-Za-z0-9\-.]{1,64}', value) else hashlib.sha256(value.encode()).hexdigest()[:32]


def _concept(value):
    return {'text': str(value or 'record')}


def _day_extension(field, day):
    return {'url': CASE_DAY, 'extension': [{'url': 'field', 'valueString': field},
                                          {'url': 'day', 'valueInteger': day}]}


def _time_path(rt, field):
    if field == 'period':
        return 'period' if rt == 'Encounter' else 'effectivePeriod'
    if field == 'effectivePeriod':
        return 'effectivePeriod' if rt in ('Observation', 'MedicationStatement') else 'context.period'
    return {'Communication': 'sent', 'MedicationRequest': 'authoredOn',
            'MedicationDispense': 'whenHandedOver',
            'Basic': 'created', 'DocumentReference': 'context.period.start'}.get(rt, 'effectiveDateTime')


def _set(obj, path, value):
    parts = path.split('.')
    for part in parts[:-1]:
        obj = obj.setdefault(part, {})
    obj[parts[-1]] = value


def _get(obj, path):
    for part in path.split('.'):
        obj = obj[part]
    return obj


def to_fhir(result, days=None, *, tool=None, request_url=None, head=None, patient_id='patient'):
    """Project a tool result into valid FHIR, preserving nonstandard source fields."""
    days = days or DayMap()
    raw = copy.deepcopy(result)
    rt = raw.get('resourceType') or (head or {}).get('entryResourceType') or 'Basic'
    if raw.get('error') or raw.get('found') is False:
        rt = 'OperationOutcome'
    elif tool == 'list_sources':
        rt = 'DocumentReference'
    elif tool == 'get_menu':
        rt = 'Bundle'
    elif tool == 'get_protocol' and rt != 'Bundle':
        rt = 'PlanDefinition'
    out = {'resourceType': rt}
    if 'id' in raw:
        out['id'] = _id(raw['id'])
    source = (raw.get('meta') or (head or {}).get('meta') or {}).get('source')
    if source:
        out['meta'] = {'source': 'http://haenv.example/fhir/source/' + quote(str(source), safe='')}
    code = raw.get('code') or (head or {}).get('code') or {}
    label = code.get('text') if isinstance(code, dict) else str(code)
    label = label or source or raw.get('name') or raw.get('target') or 'record'
    if rt == 'OperationOutcome':
        out['issue'] = [{'severity': 'error' if raw.get('error') or raw.get('ordered') is False else 'warning',
                         'code': 'invalid' if raw.get('error') else ('forbidden' if raw.get('ordered') is False else 'not-found'),
                         'diagnostics': str(raw.get('error') or raw.get('result') or 'No matching record')}]
    elif rt == 'Bundle':
        out.update(type='searchset', total=raw.get('total', len(raw.get('items', []))))
        items = raw.pop('entry', None)
        if items is None:
            items = raw.pop('items', [])
        out['entry'] = [{'resource': to_fhir(e, days, tool='menu-item' if tool == 'get_menu' else
                                           ('get_protocol' if tool == 'get_protocol' else None), head=result,
                                           patient_id=patient_id)}
                        for e in items]
        earlier = (raw.get('truncated') or {}).get('earlier_page')
        if earlier and request_url:
            parsed = urlsplit(request_url)
            params = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                      if not (k == 'date' and v.startswith('le'))]
            params.append(('date', 'le' + days.to_date(earlier['to_day'])))
            out['link'] = [{'relation': 'next', 'url': parsed.path + '?' + urlencode(params)}]
    elif rt == 'Patient':
        if raw.get('gender') in ('male', 'female', 'other', 'unknown'):
            out['gender'] = raw['gender']
    elif rt == 'Observation':
        out.update(status='unknown', code=_concept(label))
        category = raw.get('category') or (head or {}).get('category')
        if category:
            out['category'] = [{'coding': [{'system': 'http://terminology.hl7.org/CodeSystem/observation-category',
                                           'code': str(category)}]}]
        value = (raw.get('valueQuantity') or {}).get('value', raw.get('value'))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out['valueQuantity'] = {'value': value}
            if raw.get('unit'):
                out['valueQuantity']['unit'] = str(raw['unit'])
        elif value is not None:
            out['valueString'] = str(value)
    elif rt in ('MedicationStatement', 'MedicationRequest'):
        out.update(status='unknown' if rt == 'MedicationStatement' else 'active',
                   medicationCodeableConcept=_concept(raw.get('drug') or raw.get('medication') or label),
                   subject={'reference': 'Patient/' + _id(patient_id)})
        if rt == 'MedicationRequest':
            out['intent'] = 'order'
    elif rt == 'MedicationDispense':
        out.update(status='unknown', medicationCodeableConcept=_concept(raw.get('drug') or label))
    elif rt == 'Encounter':
        out.update(status='unknown', **{'class': {'system': 'http://terminology.hl7.org/CodeSystem/v3-ActCode', 'code': 'AMB'}})
    elif rt == 'Communication':
        out['status'] = 'unknown'
        out['payload'] = [{'contentString': _json(fhir.unshape(result))}]
    elif rt == 'DiagnosticReport':
        out.update(status='unknown', code=_concept(label))
    elif rt == 'DocumentReference':
        out.update(status='current', content=[{'attachment': {'contentType': 'application/json',
                                                            'data': base64.b64encode(_json(result).encode()).decode()}}])
    elif rt == 'PlanDefinition':
        out.update(status='active', title=str(raw.get('name') or label))
        if raw.get('topics'):
            out['topic'] = [_concept(t) for t in raw['topics']]
    elif tool == 'menu-item':
        out.update(resourceType='ActivityDefinition', status='active', title=str(raw.get('target') or label),
                   code=_concept(raw.get('target') or label))
        rt = 'ActivityDefinition'
    else:
        out['code'] = _concept(label)
    if rt in ('Observation', 'MedicationStatement', 'MedicationRequest', 'MedicationDispense',
              'Encounter', 'Communication', 'DiagnosticReport', 'DocumentReference', 'Basic'):
        out['subject'] = {'reference': 'Patient/' + _id(patient_id)}
    extensions = []
    for field in ('effectiveDay', 'effectivePeriod', 'period'):
        value = raw.get(field)
        # Stream Bundle heads have a prose effectivePeriod, rather than dates.
        if field == 'effectiveDay' and isinstance(value, int):
            path = _time_path(rt, field)
            if rt == 'Observation' and isinstance((head or {}).get('effectivePeriod'), str):
                path = 'effectivePeriod.start'
                _set(out, 'effectivePeriod.end', days.to_date(value))
            _set(out, path, days.to_date(value))
            extensions.append(_day_extension(field + ':' + path, value))
            raw.pop(field)
        elif isinstance(value, dict) and any(k in value for k in ('startDay', 'endDay')):
            path = _time_path(rt, field)
            for key, day in value.items():
                if key in ('startDay', 'endDay'):
                    dest = path + '.' + key[:-3]
                    _set(out, dest, days.to_date(day))
                    extensions.append(_day_extension(field + '.' + key + ':' + dest, day))
            remainder = {k: v for k, v in value.items() if k not in ('startDay', 'endDay')}
            if remainder:
                raw[field] = remainder
            else:
                raw.pop(field)
    extensions.insert(0, {'url': FIELDS, 'valueString': _json(raw)})
    if rt == 'Bundle':
        out.setdefault('meta', {})['extension'] = extensions
    else:
        out['extension'] = extensions
    return out


def from_fhir(resource, days=None):
    """Restore a tool result, verifying the calendar date against each case-day extension."""
    days = days or DayMap()
    extensions = resource.get('extension', []) + resource.get('meta', {}).get('extension', [])
    fields = next(e['valueString'] for e in extensions if e['url'] == FIELDS)
    raw = json.loads(fields)
    for ext in extensions:
        if ext['url'] != CASE_DAY:
            continue
        parts = {e['url']: e for e in ext['extension']}
        field, path = parts['field']['valueString'].split(':')
        day = days.to_day(_get(resource, path))
        if day != parts['day']['valueInteger']:
            raise ValueError('Calendar date and case-day extension disagree')
        _set(raw, field, day)
    if resource['resourceType'] == 'Bundle':
        key = 'entry' if raw.get('resourceType') == 'Bundle' else 'items'
        raw[key] = [from_fhir(e['resource'], days) for e in resource.get('entry', [])]
    return raw


def outcome(message):
    return to_fhir({'error': message})


def request_for(tool, args, patient_id='patient', days=None, count=400, *, error=False):
    """Turn rest_form into an escaped HTTP request, converting case-day bounds to dates."""
    days = days or DayMap()
    if error or tool not in fhir.REST or not isinstance(args, dict):
        return 'POST', '/$' + quote(str(tool), safe=''), {
            'resourceType': 'Parameters', 'parameter': [{'name': 'arguments', 'valueString': _json(args)}]}
    # rest_form is a display format. Encode values before using it as an HTTP
    # request so a source name containing &, + or spaces remains one argument.
    encoded = {k: quote(str(v), safe='') for k, v in args.items() if v is not None}
    for key in ('from_day', 'to_day'):
        if args.get(key) is not None:
            encoded[key] = days.to_date(int(args[key]))
    form = fhir.rest_form(tool, encoded, quote(patient_id, safe=''), count)
    method, path = form.split(' ', 2)[:2]
    parsed = urlsplit(path)
    q = parse_qsl(parsed.query, keep_blank_values=True)
    # The display helper prints absent optional topic/keyword placeholders.
    # They are not arguments of the original tool call.
    q = [(k, v) for k, v in q if not (tool == 'get_protocol' and k in ('topic', 'keyword')
                                     and args.get(k) in (None, ''))]
    if tool in ('get_series', 'get_labs') and args.get('name') == '':
        q.append(('code', ''))
    if tool == 'get_record' and 'section' in args:
        q.append(('section', str(args['section'])))
    path = parsed.path
    params = fhir.REST[tool][2]
    # Preserve malformed/extra arguments too, so replay tests the same dispatcher.
    represented = {src for _, src in params} | {'from_day', 'to_day'}
    if tool == 'get_record':
        represented.add('section')
    q.extend((k, _json(v)) for k, v in args.items() if k not in represented and method == 'GET')
    body = None
    if method == 'POST':
        body = {'resourceType': 'ServiceRequest', 'status': 'active', 'intent': 'order',
                'subject': {'reference': 'Patient/' + _id(patient_id)}, 'code': {'text': args.get('target')}}
        if set(args) != {'target'}:
            body['extension'] = [{'url': FIELDS, 'valueString': _json(args)}]
    return method, path + ('?' + urlencode(q) if q else ''), body


class FHIRServer(ThreadingHTTPServer):
    """Exactly one question session per server; concurrent clients share its lock."""
    daemon_threads = True

    def __init__(self, address, session: ToolSession, *, anchor=DEFAULT_ANCHOR):
        self.session, self.days = session, DayMap(anchor)
        super().__init__(address, FHIRHandler)

    def capability(self):
        resources = {}
        names = {s.name for s in self.session.specs()}
        for name in names:
            method, path, params = fhir.REST[name]
            if name == 'get_record':
                kinds = set((self.session.store.adapter.fhir_types or {}).values()) if self.session.store.adapter else set()
                types = kinds | {'Basic', 'Encounter', 'Observation'}
            else:
                types = {{'get_protocol': 'PlanDefinition', 'get_menu': 'ActivityDefinition',
                          'list_sources': 'DocumentReference'}.get(name, path.split('/')[1].replace('{id}', 'Patient'))}
            for rt in types:
                r = resources.setdefault(rt, {'type': rt, 'interaction': [], 'searchParam': []})
                interaction = 'create' if method == 'POST' else ('read' if name == 'get_profile' else 'search-type')
                if {'code': interaction} not in r['interaction']:
                    r['interaction'].append({'code': interaction})
                parameters = [(k, 'reference' if k == 'patient' else 'string') for k, _ in params]
                if name not in ('get_profile', 'list_sources', 'order_test'):
                    parameters += [('date', 'date'), ('_count', 'number')]
                if name == 'get_record':
                    parameters += [('section', 'string')]
                for key, typ in parameters:
                    if not any(p['name'] == key for p in r['searchParam']):
                        r['searchParam'].append({'name': key, 'type': typ})
        return {'resourceType': 'CapabilityStatement', 'status': 'active', 'date': DEFAULT_ANCHOR,
                'kind': 'instance', 'fhirVersion': '4.0.1', 'format': ['json'],
                'implementation': {'description': 'Single synthetic question session; patient: Patient/'
                                    + _id(self.session.patient_id)},
                'rest': [{'mode': 'server', 'resource': list(resources.values())}]}

    def dispatch(self, method, url, body=None):
        with self.session.lock:
            try:
                return self._dispatch(method, url, body)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                return 400, outcome(str(exc))

    def _dispatch(self, method, url, body):
        parsed = urlsplit(url)
        path = unquote(parsed.path).rstrip('/') or '/'
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        q = {}
        for key, value in pairs:
            if key != 'date' and key in q:
                raise ValueError(f'Duplicate parameter: {key}')
            if key != 'date':
                q[key] = value
        if method == 'GET' and (path == '/metadata' or (path == '/' and not pairs)):
            return 200, self.capability()
        if method == 'POST' and path == '/' and isinstance(body, dict) and body.get('type') == 'batch':
            return self._batch(body)
        if method == 'POST' and path.startswith('/$'):
            if not isinstance(body, dict) or body.get('resourceType') != 'Parameters':
                raise ValueError('A tool operation requires a Parameters resource')
            parameters = body.get('parameter', [])
            if len(parameters) != 1 or parameters[0].get('name') != 'arguments':
                raise ValueError('Expected one arguments parameter')
            result, error = self.session.call(path[2:], json.loads(parameters[0]['valueString']))
            return (429 if error == 'over_budget' else (400 if error else 200)), to_fhir(result, self.days, tool=path[2:],
                                                   patient_id=self.session.patient_id)
        args = {}
        if method == 'POST' and path == '/ServiceRequest':
            name = 'order_test'
            if not isinstance(body, dict) or body.get('resourceType') != 'ServiceRequest':
                raise ValueError('Expected a ServiceRequest')
            args = {'target': body.get('code', {}).get('text')}
            for ext in body.get('extension', []):
                if ext['url'] == FIELDS:
                    args = json.loads(ext['valueString'])
        elif method == 'GET':
            names = {s.name for s in self.session.specs()}
            patient_ids = {self.session.patient_id, _id(self.session.patient_id)}
            aliases = {'/Basic/$protocols': 'get_protocol', '/PlanDefinition': 'get_protocol',
                       '/ServiceRequest/$menu': 'get_menu', '/ActivityDefinition': 'get_menu',
                       '/DocumentReference/$sources': 'list_sources'}
            if any(path == f'/Patient/{pid}' for pid in patient_ids):
                name = 'get_profile'
            elif any(path == f'/Patient/{pid}/$everything' for pid in patient_ids):
                name = 'list_sources'
                if q.pop('_summary', None) != 'count':
                    raise ValueError('Only _summary=count is supported')
            elif path in aliases:
                name = aliases[path]
            elif path == '/Observation' and 'section' not in q:
                name = 'get_labs' if q.get('category') == 'laboratory' else 'get_series'
            elif path == '/MedicationStatement' and 'section' not in q:
                name = 'get_meds'
            elif path == '/Communication' and 'section' not in q:
                name = 'get_notes'
            elif path.startswith('/Patient/'):
                return 404, to_fhir({'found': False, 'id': path.split('/')[2]})
            elif 'get_record' in names:
                explicit_section = 'section' in q
                section = q.pop('section', path[1:])
                if not explicit_section and section not in self.session.store.sections and path[1:] in set(
                        (getattr(self.session.store.adapter, 'fhir_types', {}) or {}).values()) | {'Basic', 'Encounter'}:
                    candidates = [s for s in self.session.store.sections if
                                  (self.session.store.adapter.fhir_types.get(s) or fhir.BY_KIND.get(
                                      fhir._kind0(self.session.store.sections[s][0]), 'Basic')) == path[1:]]
                    if len(candidates) == 1:
                        section = candidates[0]
                    else:
                        raise ValueError('Select a record section with section=; see the source inventory')
                name, args = 'get_record', {'section': section}
            else:
                return 404, outcome('Unknown route')
            patient = q.pop('patient', self.session.patient_id)
            if patient not in patient_ids:
                return 404, to_fhir({'found': False, 'patient': patient})
            count = int(q.pop('_count', self.session.max_points))
            if count != self.session.max_points:
                raise ValueError(f'_count must equal the tool row cap ({self.session.max_points})')
            for key, value in q.items():
                mapped = 'name' if key == 'code' and name in ('get_series', 'get_labs') else (
                    'kind' if key == 'category' and name == 'get_notes' else key)
                if key == 'category' and name in ('get_labs', 'get_series'):
                    if value not in ('laboratory', 'activity', 'vital-signs'):
                        raise ValueError('Unsupported Observation category')
                    continue
                args[mapped] = value
            for key, value in pairs:
                if key != 'date':
                    continue
                bound = {'ge': 'from_day', 'le': 'to_day'}.get(value[:2])
                if bound is None or bound in args:
                    raise ValueError('Use one ge and one le date bound')
                # Legacy rest_form uses signed integer case days.
                args[bound] = int(value[2:]) if re.fullmatch(r'-?\d+', value[2:]) else self.days.to_day(value[2:])
        else:
            return 405, outcome('Unsupported method')
        result, error = self.session.call(name, args)
        status = 400 if error else (403 if result.get('ordered') is False else
                                   (404 if result.get('found') is False else 200))
        if error == 'over_budget':
            status = 429
        return status, to_fhir(result, self.days, tool=name, request_url=url,
                               patient_id=self.session.patient_id)

    def _batch(self, body):
        entries = body.get('entry', [])
        if not entries or any(e.get('request', {}).get('method') != 'POST' or
                              e.get('request', {}).get('url', '').lstrip('/') != 'ServiceRequest' or
                              e.get('resource', {}).get('resourceType') != 'ServiceRequest' or
                              not isinstance(e.get('resource', {}).get('code', {}).get('text'), str)
                              for e in entries):
            raise ValueError('A purchase batch contains POST ServiceRequest entries with code.text')
        if self.session.max_calls is not None and self.session.n_calls + len(entries) > self.session.max_calls:
            return 429, outcome('Tool call limit exceeded')
        targets = [e['resource']['code']['text'] for e in entries]
        results = self.session.order_tests(targets)
        self.session.n_calls += len(entries)
        return 200, {'resourceType': 'Bundle', 'type': 'batch-response', 'entry': [
            {'resource': to_fhir(r, self.days, tool='order_test', patient_id=self.session.patient_id),
             'response': {'status': '403' if r.get('ordered') is False else ('404' if r.get('found') is False else '200')}}
            for r in results]}


class FHIRHandler(BaseHTTPRequestHandler):
    """FHIR JSON only; suppress access logs (MCP also reserves stdout for its protocol)."""
    protocol_version = 'HTTP/1.1'
    disable_nagle_algorithm = True

    def log_message(self, *args):
        pass

    def send_error(self, code, message=None, explain=None):
        """Keep parser, URI-limit and unsupported-method errors in the FHIR format."""
        self.close_connection = True
        self._send(code, outcome(message or self.responses.get(code, ('Request failed',))[0]))

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            pass

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def do_PUT(self):
        self._send(405, outcome('Unsupported method'))

    do_DELETE = do_PUT
    do_PATCH = do_PUT

    def _handle(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 0 or length > 1024 * 1024:
                raise ValueError('Request body limit is 1 MiB')
            body = json.loads(self.rfile.read(length)) if length else None
        except (ValueError, UnicodeDecodeError) as exc:
            self.close_connection = True
            self._send(400, outcome(str(exc)))
            return
        self._send(*self.server.dispatch(self.command, self.path, body))

    def _send(self, status, resource):
        data = _json(resource).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/fhir+json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        if self.close_connection:
            self.send_header('Connection', 'close')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(data)
