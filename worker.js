
importScripts('/pyodide/pyodide.js');

const signedURL = (url, signature, query = {}) => {
	if (!signature)
		return url;
	const qs = new URLSearchParams(Object.assign({}, query, signature));
	return `${url}?${qs.toString()}`;
};

const pythonModuleWrapper = async (module_url, reload=false) => ({
	module: await fetch(module_url, {cache: reload ? 'no-cache' : 'default'}).then(res => res.text()),
	pyodide: await (async () => {
		try {
			const pyodide = await loadPyodide({indexURL: 'https://modelrxiv.org/pyodide/'});
			await pyodide.loadPackage('numpy'); // NumPy loaded by default, implement module definition in model builder
			return pyodide;
		} catch (e) {
			return pyodide;
		}
	})(),
	defaults: function () {
		const code = `${this.module}
result = defaults()
`;
		this.pyodide.runPython(code);
		const outputPr = this.pyodide.globals.get('result');
		const result = outputPr.toJs();
		outputPr.destroy();
		return result instanceof Map ? Object.fromEntries(result) : result;
	},
	step: function (params, _step, t) {
		const code = `${this.module}
result = step(params, _step, ${t})
`;
		this.pyodide.globals.set('params', this.pyodide.toPy(params));
		this.pyodide.globals.set('_step', this.pyodide.toPy(_step));
		this.pyodide.runPython(code);
		const outputPr = this.pyodide.globals.get('result');
		const result = outputPr.toJs();
		outputPr.destroy();
		return result instanceof Map ? Object.fromEntries(result) : result;
	},
	run: function (params) {
		const code = `${this.module}
result = run(params)
`;
		this.pyodide.globals.set('params', this.pyodide.toPy(params));
		this.pyodide.runPython(code);
		const outputPr = this.pyodide.globals.get('result');
		const result = outputPr.toJs();
		outputPr.destroy();
		return result instanceof Map ? Object.fromEntries(result) : result;
	}
});

const scriptWrapper = (script, framework) => {
	switch(framework) {
		case 'py':
			return pythonModuleWrapper(script);
		case 'js':
		default:
			return import(script);
	}
};

const scriptFromSources = (sources, credentials) => {
	const script_source = sources[0];
	const script_url = script_source.private ? signedURL(`/users/${credentials.user_id}/${script_source.model_id}.${script_source.framework}`, credentials.cdn) : `/models/${script_source.model_id}.${script_source.framework}`;
	return script_url;
};

const runParams = async (script, framework, sources, fixed_params, variable_params) => {
	const step_module = await scriptWrapper(script, framework);
	return variable_params.map(variable_params => {
		const params = Object.assign({}, fixed_params, variable_params);
		return step_module.run(params);
	});
};

const test = async (script, framework, sources) => {
	try {
		const step_module = await scriptWrapper(script, framework);
		const params = Object.assign({}, step_module.defaults());
		return {input_params: params, dynamics_params: step_module.step ? step_module.step(step_module.defaults(), undefined, 0) : {}, result_params: step_module.run(step_module.defaults())};
	} catch (e) {
		return {error: e};
	}
};

self.addEventListener("message", async e => {
	const request = e.data;
	const script = scriptFromSources(request.sources, request.credentials);
	const result = request.fixed_params.test ? await test(script, request.framework, request.sources) : await runParams(script, request.framework, request.sources, request.fixed_params, request.variable_params);
	self.postMessage(result);
});
