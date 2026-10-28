import { motion } from 'framer-motion';
import { BrainCircuit, Rocket, Network, FileSearch, Languages } from 'lucide-react';

export default function LandingPage() {
  const container = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.1 } } };
  const item = { hidden: { opacity: 0, y: 20, filter: 'blur(8px)' }, show: { opacity: 1, y: 0, filter: 'blur(0px)', transition: { type: 'spring', stiffness: 100, damping: 20 } } };

  return (
    <motion.div variants={container} initial="hidden" animate="show" className="max-w-5xl mx-auto w-full pt-24 pb-12 px-8 flex flex-col items-center">
      <motion.div variants={item} className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-2xl shadow-indigo-500/40 mb-8 animate-float">
        <BrainCircuit size={32} className="text-white" />
      </motion.div>
      
      <motion.h1 variants={item} className="text-5xl md:text-6xl font-black tracking-tight text-center mb-4 leading-tight">
        Explore source code <br/><span className="text-gradient">across languages</span>
      </motion.h1>
      
      <motion.p variants={item} className="text-slate-400 text-lg text-center max-w-2xl mb-10 leading-relaxed">
        Index a repository and ask questions in English, French, Chinese or a mix. Inspect the retrieved source locations and verify explanations against the code.
      </motion.p>
      
      <motion.div variants={item} className="flex gap-4 mb-16">
        <div className="px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-semibold flex items-center gap-2 shadow-lg shadow-indigo-500/25">
          <Rocket size={18} /> Get Started in Sidebar
        </div>
      </motion.div>

      <motion.div variants={item} className="w-full max-w-3xl glass-card p-6 mb-20 text-left">
        <h2 className="text-sm font-semibold text-slate-200 mb-4">Example questions for micrograd</h2>
        <ul className="space-y-3 text-sm text-slate-300">
          <li>Where is <code>Value.backward</code> defined?</li>
          <li>Qu'appelle <code>Value.backward</code> ?</li>
          <li><code>Value.backward</code> 调用了哪些函数？</li>
        </ul>
        <p className="mt-4 text-xs text-slate-400">Examples depend on the indexed repository. Graph references are static observations, not proof of runtime calls.</p>
      </motion.div>

      {/* Feature Grid */}
      <motion.div variants={container} className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-5xl">
        <FeatureCard icon={<Network size={20}/>} title="Source graph" desc="Explore available static caller and callee evidence." color="text-indigo-400" bg="bg-indigo-500/10" />
        <FeatureCard icon={<FileSearch size={20}/>} title="Source evidence" desc="Inspect retrieved file paths, symbols and line ranges." color="text-violet-400" bg="bg-violet-500/10" />
        <FeatureCard icon={<Languages size={20}/>} title="Multilingual questions" desc="Ask in English, French, Chinese or mixed language." color="text-emerald-400" bg="bg-emerald-500/10" />
      </motion.div>
    </motion.div>
  );
}

function FeatureCard({ icon, title, desc, color, bg }) {
  return (
    <motion.div 
      whileHover={{ y: -4, scale: 1.01 }}
      transition={{ type: "spring", stiffness: 400, damping: 30 }}
      className="glass-card p-6 relative overflow-hidden group cursor-default"
    >
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:translate-x-[200%] transition-transform duration-1000 ease-in-out"></div>
      <div className={`w-10 h-10 rounded-xl ${bg} ${color} flex items-center justify-center mb-4`}>{icon}</div>
      <h3 className="font-bold text-slate-200 mb-2">{title}</h3>
      <p className="text-sm text-slate-400 leading-relaxed">{desc}</p>
    </motion.div>
  );
}
