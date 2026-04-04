from AlgorithmImports import *
from collections import deque
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

@dataclass
class FVG:
    bar_index: int; direction: int; top: float; bottom: float; size: float
    candle2_open: float; status: str = "untested"; creation_time: datetime = None
    last_signal_idx: int = -999; invalidation_close: float = 0.0

@dataclass
class TS:
    direction: int; entry_price: float; stop_price: float; tp1_price: float
    contracts: int; entry_time: datetime; entry_bar_idx: int
    trimmed: bool = False; signal_type: str = ""; size_mult: float = 1.0
    orig_stop_dist: float = 0.0; trim_r: float = 0.0; orig_contracts: int = 0

class LantoNQStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1); self.SetEndDate(2026, 3, 26)
        self.SetCash(100000); self.SetTimeZone("America/New_York")
        self.nq = self.AddFuture(Futures.Indices.NASDAQ100EMini, Resolution.Minute,
            dataNormalizationMode=DataNormalizationMode.Raw,
            dataMappingMode=DataMappingMode.OpenInterest, contractDepthOffset=0)
        self.nq.SetFilter(0, 90); self.nq_symbol = self.nq.Symbol
        c5 = TradeBarConsolidator(timedelta(minutes=5))
        c5.DataConsolidated += self.On5m
        self.SubscriptionManager.AddConsolidator(self.nq_symbol, c5)
        c1h = TradeBarConsolidator(timedelta(hours=1))
        c1h.DataConsolidated += self.On1h
        self.SubscriptionManager.AddConsolidator(self.nq_symbol, c1h)
        c4h = TradeBarConsolidator(timedelta(hours=4))
        c4h.DataConsolidated += self.On4h
        self.SubscriptionManager.AddConsolidator(self.nq_symbol, c4h)
        self.b5 = deque(maxlen=200); self.b1h = deque(maxlen=50); self.b4h = deque(maxlen=30)
        self.atr = AverageTrueRange(14, MovingAverageType.Simple)
        self.bi = 0; self.fvgs5: List[FVG] = []; self.fvgs1h: List[FVG] = []
        self.ts: Optional[TS] = None
        self.cur_date = None; self.d_loss = 0; self.d_pnl_r = 0.0
        self.cum_r = 0.0; self.peak_r = 0.0
        self.bias = 0; self.ovn_h = 0.0; self.ovn_l = float("inf")
        self.sh: List[tuple] = []; self.sl: List[tuple] = []
        self.P = {"da":0.8,"db":0.60,"mfa":0.5,"ft":0.60,"rb":0.55,"cd":10,"ms":10,
            "nr":1000,"rr":500,"pv":20,"dml":2.0,"mcl":1,"tp":0.50,"ns":2,"nym":2.0,
            "dt1":-2.0,"ds1":0.75,"dt2":-10.0,"ds2":0.25,"emin":3,"emax":4,"ebw":0.65,"epa":0.3}
        self.ntrades = 0; self.nwins = 0; self.tot_r = 0.0
        self.last_contract = None  # track contract rolls
        self.SetWarmUp(timedelta(days=30))

    def On5m(self, s, bar):
        if self.IsWarmingUp:
            self.b5.append(bar); self.bi += 1
            tb = TradeBar(bar.Time,bar.Symbol,bar.Open,bar.High,bar.Low,bar.Close,bar.Volume,timedelta(minutes=5))
            self.atr.Update(tb); self._usw(bar); return
        self.b5.append(bar); self.bi += 1
        # Detect contract rollover — clear all FVGs and swings (price levels changed)
        cur_contract = str(bar.Symbol) if hasattr(bar,'Symbol') else None
        if cur_contract and self.last_contract and cur_contract != self.last_contract:
            self.fvgs5.clear(); self.fvgs1h.clear()
            self.sh.clear(); self.sl.clear()
            self.b5.clear(); self.b5.append(bar)
            self.Debug(f"ROLL: {self.last_contract} -> {cur_contract}, cleared FVGs/swings")
        self.last_contract = cur_contract
        tb = TradeBar(bar.Time,bar.Symbol,bar.Open,bar.High,bar.Low,bar.Close,bar.Volume,timedelta(minutes=5))
        self.atr.Update(tb)
        if not self.atr.IsReady: return
        av = self.atr.Current.Value
        et = bar.EndTime; hf = et.hour + et.minute/60.0
        ss = "ny" if 9.5<=hf<16 else ("london" if 3<=hf<9.5 else "asia")
        self._nd(et)
        if ss in ("asia","london"):
            self.ovn_h = max(self.ovn_h, bar.High); self.ovn_l = min(self.ovn_l, bar.Low)
        self._usw(bar); self._ufvg(bar, av)
        if self.ts is not None: self._mgr(bar, av); return
        if 9.5<=hf<10: return
        sig = self._csig(bar, av, ss)
        if sig and self._filt(sig, bar, ss, av): self._enter(sig, bar, ss, av)

    def On1h(self, s, bar):
        self.b1h.append(bar)
        if len(self.b1h)>=3: self._htf(self.b1h, self.fvgs1h)

    def On4h(self, s, bar):
        self.b4h.append(bar)
        if len(self.b4h)<3: return
        b = list(self.b4h); l = b[-1]; body = abs(l.Close-l.Open)
        ae = sum(x.High-x.Low for x in b[-5:])/min(5,len(b[-5:]))
        if body > 0.5*ae: self.bias = 1 if l.Close>l.Open else -1

    def _gs(self, hf):
        return "ny" if 9.5<=hf<16 else ("london" if 3<=hf<9.5 else "asia")

    def _nd(self, et):
        td = et.date()
        if et.hour>=18: td = et.date()+timedelta(days=1)
        if td != self.cur_date:
            self.cur_date=td; self.d_loss=0; self.d_pnl_r=0.0
            self.ovn_h=0.0; self.ovn_l=float("inf")

    def _ufvg(self, bar, av):
        if len(self.b5)<3: return
        bs = list(self.b5); c1,c2,c3 = bs[-3],bs[-2],bs[-1]
        for dr,gap,bot,top in [(1,c3.Low-c1.High,c1.High,c3.Low),(-1,c1.Low-c3.High,c3.High,c1.Low)]:
            if (dr==1 and c1.High<c3.Low) or (dr==-1 and c1.Low>c3.High):
                sz = abs(gap)
                if sz >= self.P["mfa"]*av:
                    bd = abs(c2.Close-c2.Open); rn = c2.High-c2.Low
                    if rn>0 and bd/rn>=self.P["db"] and bd>=self.P["da"]*av:
                        self.fvgs5.append(FVG(self.bi-1,dr,top,bot,sz,c2.Open,creation_time=c2.EndTime))
        for f in self.fvgs5[:]:
            if f.status=="invalidated": continue
            if f.direction==1:
                if bar.Low<=f.top and f.status=="untested": f.status="tested_rejected"
                if bar.Close<f.bottom:
                    f.status="invalidated"; f.invalidation_close=bar.Close
            else:
                if bar.High>=f.bottom and f.status=="untested": f.status="tested_rejected"
                if bar.Close>f.top:
                    f.status="invalidated"; f.invalidation_close=bar.Close
        self.fvgs5 = [f for f in self.fvgs5 if self.bi-f.bar_index<500]

    def _htf(self, bd, fl):
        bs = list(bd); c1,c2,c3 = bs[-3],bs[-2],bs[-1]
        if c1.High<c3.Low:
            fl.append(FVG(self.bi,1,c3.Low,c1.High,c3.Low-c1.High,c2.Open,creation_time=c2.EndTime))
        if c1.Low>c3.High:
            fl.append(FVG(self.bi,-1,c1.Low,c3.High,c1.Low-c3.High,c2.Open,creation_time=c2.EndTime))
        fl[:] = [f for f in fl if self.bi-f.bar_index<2000]

    def _usw(self, bar):
        if len(self.b5)<5: return
        bs = list(self.b5); cd = bs[-2]; lb = bs[-5:-2]
        if all(cd.High>b.High for b in lb) and cd.High>bs[-1].High:
            self.sh.append((self.bi-1,cd.High))
            if len(self.sh)>50: self.sh.pop(0)
        if all(cd.Low<b.Low for b in lb) and cd.Low<bs[-1].Low:
            self.sl.append((self.bi-1,cd.Low))
            if len(self.sl)>50: self.sl.pop(0)

    def _csig(self, bar, av, ss):
        if len(self.b5)<10: return None
        sig = self._ctrend(bar, av)
        return sig if sig else self._cmss(bar, av)

    def _ctrend(self, bar, av):
        cur = bar; prev = list(self.b5)[-2]
        for f in self.fvgs5:
            if f.status!="tested_rejected": continue
            if self.bi-f.last_signal_idx<self.P["cd"]: continue
            if self.bi-f.bar_index<2: continue
            d = f.direction
            test = (prev.Low<=f.top and cur.Close>f.bottom) if d==1 else (prev.High>=f.bottom and cur.Close<f.top)
            if not test: continue
            bd = abs(cur.Close-cur.Open); rn = cur.High-cur.Low
            if rn<=0 or bd/rn<self.P["rb"]: continue
            good_close = (cur.Close>cur.Open) if d==1 else (cur.Close<cur.Open)
            if not good_close: continue
            if not self._flu(d): continue
            f.last_signal_idx = self.bi
            return {"direction":d,"type":"trend","fvg":f,"entry_price":cur.Close,
                    "model_stop":f.candle2_open,"tp1":self._irl(d,cur.Close)}
        return None

    def _cmss(self, bar, av):
        bs = list(self.b5); cur = bar
        if len(bs)<20: return None
        for f in self.fvgs5:
            if f.status!="invalidated": continue
            if self.bi-f.bar_index>200: continue
            idir = -f.direction
            inv_body = 0
            for j in range(max(0,len(bs)-50),len(bs)):
                b = bs[j]
                if (f.direction==1 and b.Close<f.bottom) or (f.direction==-1 and b.Close>f.top):
                    inv_body = abs(b.Close-b.Open); break
            if inv_body < self.P["da"]*av*0.5: continue
            swept = False
            for k in range(max(0,len(bs)-30),len(bs)):
                b = bs[k]
                if idir==1:
                    for sl in self.sl:
                        if b.Low<sl[1]: swept=True; break
                else:
                    for sh in self.sh:
                        if b.High>sh[1]: swept=True; break
                if swept: break
            if not swept: continue
            if idir==1:
                rt = cur.Low<=f.top and cur.Close>f.bottom and cur.Close>cur.Open
            else:
                rt = cur.High>=f.bottom and cur.Close<f.top and cur.Close<cur.Open
            if not rt: continue
            bd = abs(cur.Close-cur.Open); rn = cur.High-cur.Low
            if rn<=0 or bd/rn<self.P["rb"]: continue
            if not self._flu(idir): continue
            stop = f.bottom if idir==1 else f.top
            entry = cur.Close
            if (idir==1 and stop>=entry) or (idir==-1 and stop<=entry): continue
            f.status = "used"
            return {"direction":idir,"type":"mss","fvg":f,"entry_price":entry,
                    "model_stop":stop,"tp1":self._irl(idir,entry)}
        return None

    def _flu(self, d):
        if len(self.b5)<6: return False
        bs = list(self.b5)[-6:]
        sd = sum(1 for b in bs if (b.Close>b.Open)==(d>0))
        brs = [abs(b.Close-b.Open)/(b.High-b.Low) for b in bs if b.High-b.Low>0]
        return (0.4*sd/6+0.3*(np.mean(brs) if brs else 0)+0.3*0.5) >= self.P["ft"]

    def _irl(self, d, entry):
        if d==1:
            c = [s[1] for s in self.sh if s[1]>entry]
            return min(c) if c else entry+20
        else:
            c = [s[1] for s in self.sl if s[1]<entry]
            return max(c) if c else entry-20

    def _filt(self, sig, bar, ss, av):
        d=sig["direction"]; e=sig["entry_price"]; st=sig["model_stop"]
        if ss=="ny" and d!=1: return False
        if ss=="london" and d!=-1: return False
        if ss=="asia": return False
        if self.bias!=0 and d==-self.bias: return False
        sd = abs(e-st)
        if sd<self.P["ms"]: return False
        if (d==1 and st>=e) or (d==-1 and st<=e): return False
        if self.d_loss>=self.P["mcl"]: return False
        if self.d_pnl_r<=-self.P["dml"]: return False
        if self.ts is not None: return False
        return True

    def _enter(self, sig, bar, ss, av):
        d=sig["direction"]; st=sig["model_stop"]; ee=bar.Close
        sd = abs(ee-st)
        dd = self.cum_r-self.peak_r
        sm = self.P["ds2"] if dd<=self.P["dt2"] else (self.P["ds1"] if dd<=self.P["dt1"] else 1.0)
        dow = bar.EndTime.weekday()
        rd = (self.P["nr"] if dow in(1,2,3) else self.P["rr"])*sm
        ct = max(1,int(rd/(sd*self.P["pv"])))
        tp = sig["tp1"]
        if ss=="ny":
            td = abs(tp-ee)
            tp = ee+td*self.P["nym"] if d==1 else ee-td*self.P["nym"]
        ms = self._gms()
        if ms is None: return
        tk = self.MarketOrder(ms, ct if d==1 else -ct)
        fp = tk.AverageFillPrice if tk.Status==OrderStatus.Filled else ee
        self.ts = TS(d,fp,st,tp,ct,bar.EndTime,self.bi,signal_type=sig["type"],size_mult=sm,
                     orig_stop_dist=sd,orig_contracts=ct)

    def _gms(self):
        for ch in self.CurrentSlice.FutureChains:
            if ch.Key==self.nq_symbol:
                cs = sorted([c for c in ch.Value], key=lambda c:c.Expiry)
                if cs: return cs[0].Symbol
        return self.nq.Mapped if hasattr(self.nq,'Mapped') else None

    def _mgr(self, bar, av):
        t = self.ts
        if t is None: return
        bit = self.bi-t.entry_bar_idx
        if t.direction==1:
            if bar.Low<=t.stop_price: self._exit(bar,t.stop_price,"stop"); return
        else:
            if bar.High>=t.stop_price: self._exit(bar,t.stop_price,"stop"); return
        if not t.trimmed:
            ht = (t.direction==1 and bar.High>=t.tp1_price) or (t.direction==-1 and bar.Low<=t.tp1_price)
            if ht: self._trim(bar,t); return
        if not t.trimmed and self.P["emin"]<=bit<=self.P["emax"]:
            if self._bpa(bar,t,bit,av): self._exit(bar,bar.Close,"early_cut"); return
        if t.trimmed and bit>1:
            nt = self._trail(t)
            if nt:
                if t.direction==1 and nt>t.stop_price: t.stop_price=nt
                elif t.direction==-1 and nt<t.stop_price: t.stop_price=nt
        if bar.EndTime.hour>=16: self._exit(bar,bar.Close,"eod")

    def _trim(self, bar, t):
        ms = self._gms()
        if ms is None: return
        tq = max(1,t.contracts//2)
        self.MarketOrder(ms, -tq if t.direction==1 else tq)
        # Trim R: TP1 profit / original stop distance, scaled by trim ratio
        osd = t.orig_stop_dist if t.orig_stop_dist > 0 else 1
        tp_dist = abs(t.tp1_price - t.entry_price)
        trim_ratio = tq / t.orig_contracts if t.orig_contracts > 0 else 0.5
        t.trim_r = (tp_dist / osd) * trim_ratio
        t.trimmed = True; t.contracts -= tq; t.stop_price = t.entry_price

    def _bpa(self, bar, t, bi, av):
        if len(self.b5)<bi+1: return False
        rc = list(self.b5)[-bi:]
        wr=[]; fv=0
        for b in rc:
            rn=b.High-b.Low
            if rn>0:
                mw=max(b.High-max(b.Open,b.Close),min(b.Open,b.Close)-b.Low)
                wr.append(mw/rn)
                if (t.direction==1 and b.Close>b.Open) or (t.direction==-1 and b.Close<b.Open): fv+=1
        if not wr: return False
        aw=np.mean(wr)
        pg = ((bar.Close-t.entry_price) if t.direction==1 else (t.entry_price-bar.Close))/av
        return aw>self.P["ebw"] and pg<self.P["epa"] and bi>=self.P["emin"]

    def _trail(self, t):
        n=self.P["ns"]
        if t.direction==1:
            v=[s for s in self.sl if s[0]>t.entry_bar_idx and s[1]>t.entry_price]
            return v[-n][1] if len(v)>=n else None
        else:
            v=[s for s in self.sh if s[0]>t.entry_bar_idx and s[1]<t.entry_price]
            return v[-n][1] if len(v)>=n else None

    def _exit(self, bar, ep, reason):
        t=self.ts
        if t is None: return
        ms=self._gms()
        if ms: self.Liquidate(ms)
        osd = t.orig_stop_dist if t.orig_stop_dist > 0 else 1
        pp = (ep-t.entry_price) if t.direction==1 else (t.entry_price-ep)
        if t.trimmed:
            # Remaining portion R: pts moved / original stop dist, scaled by remaining ratio
            remain_ratio = t.contracts / t.orig_contracts if t.orig_contracts > 0 else 0.5
            remaining_r = (pp / osd) * remain_ratio
            rm = t.trim_r + remaining_r
            reason_tag = "be_sweep" if abs(pp) < 1 else reason
        else:
            rm = pp / osd
            reason_tag = reason
        rms = rm * t.size_mult
        self.cum_r+=rms; self.peak_r=max(self.peak_r,self.cum_r)
        self.d_pnl_r+=rms; self.tot_r+=rms; self.ntrades+=1
        if rm<0: self.d_loss+=1
        if rm>0: self.nwins+=1
        self.Debug(f"[{reason_tag}] d={t.direction} e={t.entry_price:.1f} x={ep:.1f} R={rm:.2f} sz={rms:.2f} cum={self.cum_r:.1f}")
        self.ts=None

    def OnEndOfAlgorithm(self):
        wr=self.nwins/self.ntrades*100 if self.ntrades>0 else 0
        mdd=self.peak_r-self.cum_r if self.peak_r>self.cum_r else 0
        ppdd=self.tot_r/mdd if mdd>0.01 else 0
        self.Debug(f"=== FINAL ===")
        self.Debug(f"Trades={self.ntrades}, Wins={self.nwins}, WR={wr:.1f}%")
        self.Debug(f"TotalR={self.tot_r:.1f}, PeakR={self.peak_r:.1f}, MaxDD={mdd:.1f}R")
        self.Debug(f"PPDD={ppdd:.1f}, CumR={self.cum_r:.1f}")
        self.Debug(f"Target: 1010 trades, 203R, PPDD~5")
