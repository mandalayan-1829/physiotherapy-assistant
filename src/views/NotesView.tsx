import { useState } from 'react';
import { 
  ClipboardList, 
  Plus, 
  Trash2, 
  Tag, 
  Calendar, 
  ChevronDown, 
  ChevronRight,
  X
} from 'lucide-react';
import { Note } from '../types';

interface NotesViewProps {
  notes: Note[];
  onAddNote: (text: string, category: Note['category']) => void;
  onDeleteNote: (id: number) => void;
}

export function NotesView({ notes, onAddNote, onDeleteNote }: NotesViewProps) {
  const [noteText, setNoteText] = useState<string>('');
  const [category, setCategory] = useState<Note['category']>('clinical');
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [showAdd, setShowAdd] = useState<boolean>(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!noteText.trim()) return;
    onAddNote(noteText.trim(), category);
    setNoteText('');
    setShowAdd(false);
  };

  const filteredNotes = filterCategory === 'all'
    ? notes
    : notes.filter((n) => n.category === filterCategory);

  const getBadgeStyle = (cat: Note['category']) => {
    switch (cat) {
      case 'clinical':
        return 'bg-blue-50 text-blue-700 border-blue-200';
      case 'symptom':
        return 'bg-amber-50 text-amber-800 border-amber-200';
      case 'exercise':
        return 'bg-[#ECFDF3] text-[#065F46] border-[#A7F3D0]';
      default:
        return 'bg-slate-100 text-slate-700 border-slate-200';
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* Header */}
      <div className="pb-5 border-b border-slate-200">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-600" />
              <span className="text-xs font-mono uppercase tracking-wider text-slate-500">
                Clinical Logs & Symptoms
              </span>
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight mt-1 flex items-center gap-2">
              <ClipboardList className="w-5 h-5 text-blue-600" />
              <span>Physiotherapy Notes & Symptom Journal</span>
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1">
              Document therapist advice, range of motion observations, pain spikes, and recovery milestones.
            </p>
          </div>

          <button
            onClick={() => setShowAdd(!showAdd)}
            className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold flex items-center gap-2 transition-colors cursor-pointer self-start sm:self-auto shadow-xs"
          >
            <Plus className="w-4 h-4" />
            <span>{showAdd ? 'Close Entry Form' : 'New Journal Entry'}</span>
          </button>
        </div>
      </div>

      {/* Expandable Add Note Form */}
      {showAdd && (
        <form onSubmit={handleSubmit} className="p-4 rounded-lg bg-slate-50 border border-slate-200 space-y-3.5 text-xs">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-slate-900">Record Clinical Journal Entry</h3>
            <div className="flex items-center gap-2">
              <span className="text-slate-600 text-xs">Category:</span>
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value as Note['category'])}
                className="px-2.5 py-1 bg-white border border-slate-200 rounded-lg text-xs text-slate-800 focus:outline-none focus:border-blue-500 cursor-pointer"
              >
                <option value="clinical">Clinical Advice</option>
                <option value="symptom">Symptom / Pain Log</option>
                <option value="exercise">Exercise Observation</option>
                <option value="general">General</option>
              </select>
            </div>
          </div>

          <textarea
            required
            rows={3}
            value={noteText}
            onChange={(e) => setNoteText(e.target.value)}
            placeholder="Document clinical feedback, joint stiffness, pain rating, or doctor notes..."
            className="w-full p-3 bg-white border border-slate-200 rounded-lg text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 resize-none"
          />

          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setShowAdd(false)}
              className="px-3 py-1.5 rounded-lg bg-white border border-slate-200 text-slate-600 text-xs font-semibold cursor-pointer hover:bg-slate-50 hover:text-slate-900"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold transition-colors cursor-pointer shadow-xs"
            >
              Save Note
            </button>
          </div>
        </form>
      )}

      {/* Category Filter Line */}
      <div className="flex items-center gap-2 border-b border-slate-200 pb-3">
        {['all', 'clinical', 'symptom', 'exercise', 'general'].map((cat) => (
          <button
            key={cat}
            onClick={() => setFilterCategory(cat)}
            className={`px-3 py-1 rounded-md text-xs font-medium capitalize transition-colors cursor-pointer ${
              filterCategory === cat
                ? 'bg-slate-900 text-white border border-slate-900'
                : 'text-slate-500 hover:text-slate-900 hover:bg-slate-100'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Notes List (Clean Divider-Based List, No large cards) */}
      {filteredNotes.length === 0 ? (
        <div className="py-8 text-center text-xs text-slate-500 border border-slate-200 rounded-lg bg-slate-50">
          No entries recorded under this category.
        </div>
      ) : (
        <div className="divide-y divide-slate-200 border-t border-b border-slate-200">
          {filteredNotes.map((n) => (
            <div
              key={n.id}
              className="py-4 px-2 flex flex-col sm:flex-row sm:items-start justify-between gap-3 text-xs hover:bg-slate-50 transition-colors"
            >
              <div className="flex-1 min-w-0 pr-4 space-y-1.5">
                <div className="flex items-center gap-2.5">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider border ${getBadgeStyle(n.category)}`}>
                    {n.category}
                  </span>
                  <div className="flex items-center gap-1 text-[11px] text-slate-500 font-mono">
                    <Calendar className="w-3 h-3 text-slate-400" />
                    <span>{n.date}</span>
                  </div>
                </div>

                <p className="text-slate-800 leading-relaxed whitespace-pre-wrap text-xs sm:text-sm">
                  {n.noteText}
                </p>
              </div>

              <div className="shrink-0 self-end sm:self-start pt-1">
                <button
                  onClick={() => onDeleteNote(n.id)}
                  className="p-1.5 rounded text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors cursor-pointer"
                  title="Delete Note"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

    </div>
  );
}
