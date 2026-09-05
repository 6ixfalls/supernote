export default {
    props: {
        request: { type: Object, required: true },
        isSubmitting: { type: Boolean, default: false },
        error: { type: String, default: null }
    },
    emits: ['decision'],
    template: `
    <div class="max-w-md mx-auto bg-white rounded-3xl border border-slate-200 shadow-xl overflow-hidden mt-20">
        <div class="p-8 sm:p-12">
            <div class="text-center mb-8">
                <div class="w-16 h-16 bg-indigo-600 rounded-2xl flex items-center justify-center text-white text-2xl font-bold shadow-lg shadow-indigo-200 mx-auto mb-4">S</div>
                <h2 class="text-2xl font-bold text-slate-900">Authorize access</h2>
                <p class="text-slate-500 mt-2 break-all">{{ request.client_id }}</p>
            </div>

            <p class="text-sm text-slate-600 mb-3">This application is requesting:</p>
            <ul class="mb-8 space-y-2">
                <li v-for="scope in request.scopes" :key="scope"
                    class="rounded-xl bg-slate-50 border border-slate-200 px-4 py-3 text-sm font-medium text-slate-700">
                    {{ scope }}
                </li>
            </ul>

            <p v-if="error" class="mb-4 rounded-xl bg-red-50 px-4 py-3 text-sm text-red-600">{{ error }}</p>

            <div class="flex gap-3">
                <button type="button" @click="$emit('decision', 'deny')" :disabled="isSubmitting"
                    class="flex-1 border border-slate-200 text-slate-700 font-bold py-3 px-4 rounded-xl hover:bg-slate-50 disabled:opacity-50">
                    Deny
                </button>
                <button type="button" @click="$emit('decision', 'approve')" :disabled="isSubmitting"
                    class="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-xl shadow-lg shadow-indigo-200 disabled:opacity-50">
                    {{ isSubmitting ? 'Submitting...' : 'Approve' }}
                </button>
            </div>
        </div>
    </div>
    `
};
