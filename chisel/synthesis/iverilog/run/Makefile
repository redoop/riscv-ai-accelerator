DESIGN := soc_tb

ip_sel :=
prog :=
wave :=

TEST_MARCO := $(addprefix +, $(ip_sel), $(prog) $(wave))

SIM_TOOL  := vcs
WAVE_TOOL := verdi
WAVE_FILE := ${DESIGN}.fsdb

COMP_LOG      := -l compile.log
SIM_LOG       := -l sim.log

## filelist
RTL_FLIST := -f ../filelist/asic_top.f \
			 -f ../filelist/ip.f \
			 -f ../filelist/lib.f \
			 -f ../filelist/soc.f

SIM_INC := +incdir+../top \
		   +incdir+../utils \
		   +incdir+../../project_1854/source \
		   +incdir+../../project_1884/source \
		   +incdir+../../project_1839/source \
		   +incdir+../../project_1883/source \
		   +incdir+/nfs/share/home/qiming/soc/ysyxSoC/new_soc \
		   +incdir+/nfs/share/home/qiming/soc/ysyxSoC/new_soc/ysyxSoC/perip/spi/rtl \
		   +incdir+/nfs/share/home/qiming/soc/ysyxSoC/new_soc/ysyxSoC/perip/uart16550/rtl \
		   +incdir+../tb \
		   +incdir+../tb/include \

TB_FLIST := -f ../filelist/asic_tblist.f \
			-top soc_tb \

## vcs option
SIM_OPTIONS := -full64  +v2k -sverilog -timescale=1ns/10ps \
			   ${EXTRA} \
			   -kdb \
			   -debug_acc+all \
			   +error+500 \
			   +vcs+flush+all \
			   +lint=TFIPC-L \
			   +define+no_warning \
			   +define+S50 \
			   +define+SVA_OFF \
			   -work DEFAULT \
			   +define+RANDOMIZE_REG_INIT \
			   +define+PDK_BEHAV \
			   ${SIM_INC} 
#               -lSDL2

TIME_OPTION := +notimingcheck \
			   +nospecify \

comp:
	${SIM_TOOL} ${SIM_OPTIONS} ${TIME_OPTION} ${RTL_FLIST} ${TB_FLIST} ${COMP_LOG}

sim: comp
	./simv ${SIM_LOG} $(TEST_MARCO)

wave:
	${WAVE_TOOL} -ssf ${WAVE_FILE} -nologo

.PHONY : clean

clean :
	rm -fr csrc *simv.daidir* *simv* ucli.key vcdplus.vpd DVEfiles INCA_libs *.fsdb* *.log novas.*
